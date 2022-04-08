
import random, io
from copy import deepcopy as copy
from collections.abc import Iterable
from tqdm import tqdm
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from scipy.interpolate import interp1d

class ParameterRange():
	def __init__(self, lo, hi):
		self.lo = lo
		self.hi = hi

	def get_lower_bound(self):
		return self.lo

	def get_upper_bound(self):
		return self.hi

class BehaviorRange():
	@classmethod
	def simple_interval(cls, lo, hi):
		return cls([(lo, hi)])

	def __init__(self, lo_hi):
		assert len(lo_hi[0]) == 2, 'Incorrect format of lo_hi: ' + str(lo_hi)
		self.lo_hi = sorted(lo_hi)

	def intersection(self, other):
		i = j = 0
		n = len(self.lo_hi)
		m = len(other.lo_hi)
		intervals = []
		# Loop through all intervals unless one
		# of the interval gets exhausted
		while i < n and j < m:
			# Left bound for intersecting segment
			l = max(self.lo_hi[i][0], other.lo_hi[j][0])
			# Right bound for intersecting segment
			r = min(self.lo_hi[i][1], other.lo_hi[j][1])
			# If segment is valid print it
			if l <= r:
				intervals.append([l, r])
			# If i-th interval's right bound is
			# smaller increment i else increment j
			if self.lo_hi[i][1] < other.lo_hi[j][1]:
				i += 1
			else:
				j += 1
		return BehaviorRange(intervals)

	def union(self, other):
		intervals = self.lo_hi + other.lo_hi
		# Sorting based on the increasing order
		# of the start intervals
		intervals.sort(key = lambda x: x[0])
		# array to hold the merged intervals
		union_intervals = []
		s = float('-inf')
		max = float('-inf')
		for i in range(len(intervals)):
			a = intervals[i]
			if a[0] > max:
				if i != 0:
					union_intervals.append([s,max])
				max = a[1]
				s = a[0]
			else:
				if a[1] >= max:
					max = a[1]
		# 'max' value gives the last point of that particular interval
		# 's' gives the starting point of that interval
		# 'union_intervals' array contains the list of all merged intervals
		if max != float('-inf') and [s, max] not in union_intervals:
			union_intervals.append([s, max])
		return BehaviorRange(intervals)

	def contains(self, val):
		return any(l <= val <= h for l, h in self.lo_hi)

class Node():
	pass

class CompositionNode(Node):
	def __init__(self, child1, child2, mode):
		assert isinstance(child1, Node) and isinstance(child2, Node)
		assert mode in ['&', '>']
		self.child1 = child1
		self.child2 = child2
		self.mode = mode

	def get_formula(self):
		return f'({self.child1.get_formula()} {self.mode} {self.child2.get_formula()})'

	def get_a_b_func_value(self, u):
		# return a_func_value, b_func_value, and a tuple of effective rules,
		# which denotes the list of rules that is reponsible for the decision
		a1, b1, rs1 = self.child1.get_a_b_func_value(u)
		a2, b2, rs2 = self.child2.get_a_b_func_value(u)
		if self.mode == '&':
			if a1 and a2:
				return True, b1.intersection(b2), rs1 + rs2
			elif a1 and (not a2):
				return True, b1, rs1
			elif (not a1) and a2:
				return True, b2, rs2
			elif (not a1) and (not a2):
				return False, b1.union(b2), tuple()
		elif self.mode == '>':
			if a1:
				return True, b1, rs1
			elif (not a1) and a2:
				return True, b2, rs2
			elif (not a1) and (not a2):
				return False, b1.union(b2), tuple()

class LeafNode(Node):
	def __init__(self, rule):
		self.rule = rule

	def get_formula(self):
		return f'R{self.rule.idx}'

	def get_a_b_func_value(self, u):
		# return a_func_value, b_func_value, and a tuple of effective rules,
		# which denotes the list of rules that is reponsible for the decision
		a, b = self.rule.get_a_b_func_value(u)
		if a:
			responsible = (self.rule.idx, )
		else:
			responsible = tuple()
		return a, b, responsible

class Parameter():
	def __init__(self, name, param_range, default_value):
		assert isinstance(param_range, ParameterRange)
		self.name = name
		self.param_range = param_range
		self.default_value = default_value
		self.current_value = default_value

class Rule():
	def __init__(self, data):
		if len(data) == 6:
			self.idx, self.name, self.a_func, self.a_params, \
				self.b_func, self.b_params = data
			self.mode = 'separate'
		elif len(data) == 5:
			self.idx, self.name, self.ab_func, self.a_params, self.b_params = data
			self.mode = 'combined'
		else:
			raise Exception('Unrecognized rule data format')

	def get_params(self, func_type):
		assert func_type in ['a', 'b']
		if func_type == 'a':
			return self.a_params
		else:
			return self.b_params

	def resolve_a_params(self):
		return [p.current_value for p in self.a_params]

	def resolve_b_params(self):
		return [p.current_value for p in self.b_params]

	def get_a_b_func_value(self, u):
		if self.mode == 'separate':
			a = self.a_func(u, self.resolve_a_params())
			b = self.b_func(u, self.resolve_b_params())
		else:
			a, b = self.ab_func(u, self.resolve_a_params(), self.resolve_b_params())
		return a, b

class RuleUnion():
	def __init__(self, rules, composition_structure):
		'''
		rules is a list of constituent rules, each specified as a tuple of
		(
		 rule_idx, rule_name,
		 a_func, list of (a_param_name, a_param_range, a_param_value),
		 b_func, list of (b_param_name, b_param_range, b_param_value)
		)
		rule_idx is integer, all the names are strings, a_func and b_func are python functions
		param_range is a ParameterRange object, and param_value is a number (integer or float).
		composition structure is either an integer or a tuple specified in the following format:
		((1, '&', 2), '>', ((3, '&', 1), '>', 4))
		where everything are tuples, integers (for rule index), or strings (for composition mode)
		'''
		assert isinstance(composition_structure, int) or isinstance(composition_structure, tuple)
		assert isinstance(rules, list)
		self.rules = dict()
		for r in rules:
			assert r.idx not in self.rules, f'Duplicate rule {r.idx} not allowed. '
			self.rules[r.idx] = r
		self.composition_structure = composition_structure
		if isinstance(composition_structure, int):  # just a single rule
			self.root = self.get_leaf_node(composition_structure)
		else:
			self.root = self.setup_rule_union(composition_structure)

	def get_leaf_node(self, idx):
		return LeafNode(self.rules[idx])

	def setup_rule_union(self, substructure):
		assert len(substructure) == 3
		left, mode, right = substructure
		assert left != right
		if isinstance(left, tuple):
			left = self.setup_rule_union(left)
		else:
			left = self.get_leaf_node(left)
		if isinstance(right, tuple):
			right = self.setup_rule_union(right)
		else:
			right = self.get_leaf_node(right)
		return CompositionNode(left, right, mode)

	def get_formula(self):
		formula = self.root.get_formula()
		if formula[0] == '(':
			formula = formula[1:-1]
		return formula

	def get_rules(self):
		return self.rules

	def get_rule_by_idx(self, idx):
		return self.rules[idx]

	def get_structure_without_rule(self, substructure, idx):
		# print(substructure)
		left, mode, right = substructure
		if isinstance(left, int):
			if left == idx:
				if isinstance(right, int):
					assert right != idx
					return right
				else:
					return self.get_structure_without_rule(right, idx)
			else:
				new_left = left
		else:
			new_left = self.get_structure_without_rule(left, idx)
		if isinstance(right, int):
			if right == idx:
				if isinstance(left, int):
					assert left != idx
					return left
				else:
					return self.get_structure_without_rule(left, idx)
			else:
				new_right = right
		else:
			new_right = self.get_structure_without_rule(right, idx)
		return (new_left, mode, new_right)

	def get_rule_union_without_rule(self, idx):
		'''return a RuleUnion object with the specified rule removed'''
		if isinstance(self.composition_structure, int):
			return RuleUnionNull()
		new_rules = [r for r in self.rules.values() if r.idx != idx]
		new_structure = self.get_structure_without_rule(self.composition_structure, idx)
		return RuleUnion(new_rules, new_structure)

	def get_rule_union_for_rule(self, idx):
		'''return a RuleUnion object with only the specified rule'''
		return RuleUnion([self.rules[idx]], idx)

	def get_a_b_func_value(self, u):
		return self.root.get_a_b_func_value(u)

class RuleUnionNull():
	def get_formula(self):
		return '(null)'

class FEU():
	def __init__(self, context, idx):
		assert isinstance(context, SentenceGroupedFEU)
		self.context = context
		self.idx = idx
		self.word = context.words[idx]
		self.feature = context.features[idx]
		self.explanation = context.explanations[idx]
		self.true_label = context.true_label
		self.prediction = context.prediction
		self.L = len(context.words)

class SentenceGroupedFEU():
	'''
	a data structure to represent a sentence
	words is a list of words
	features is a list of (pre-computed) feature tuples for the words
	explanations is a list of local explanation values for the words
	true_label is 0 or 1. prediction is a float between 0 and 1.
	'''
	def __init__(self, words, features, explanations, true_label, prediction):
		assert len(words) == len(features) == len(explanations)
		self.words = words
		self.features = features
		self.explanations = explanations
		self.true_label = true_label
		self.prediction = prediction
		self.L = len(words)

	def get_random_FEU(self):
		idx = random.randint(0, self.L - 1)
		return FEU(self, idx)

	def get_all_FEUs(self):
		for idx in range(self.L):
			yield FEU(self, idx)

class Data():
	def __init__(self, sentences, exp_measure, normalize=False):
		self.exp_measure = exp_measure
		if normalize:
			factor = max(abs(self.exp_measure.cdf_min), abs(self.exp_measure.cdf_max))
			new_sentences = []
			for s in sentences:
				new_sentences.append(SentenceGroupedFEU(s.words, s.features,
					[e / factor for e in s.explanations], s.true_label, s.prediction))
			self.sentences = new_sentences
			self.exp_measure.set_factor(factor)
		else:
			self.sentences = sentences
		self.a_order_idxs = list(range(len(sentences)))
		self.b_order_idxs = list(range(len(sentences)))

	def shuffle(self, typ):
		assert typ in ['a', 'b']
		if typ == 'a':
			random.seed(0)
			random.shuffle(self.a_order_idxs)
		elif typ == 'b':
			random.seed(0)
			random.shuffle(self.b_order_idxs)

class Measure():
	def __init__(self, explanations, weights, zero_discrete):
		self.factor = 1
		self.zero_discrete = zero_discrete
		if zero_discrete:
			zero_mask = (explanations == 0)
			self.zero_pmf = sum(zero_mask * weights) / sum(weights)
			explanations = explanations[~zero_mask]
			weights = weights[~zero_mask]
		else:
			self.zero_pmf = 0
		weights = weights / sum(weights)
		kernel = gaussian_kde(explanations, weights=weights)
		density_kernel = lambda x: kernel(x)[0]
		self.cdf_min = min(explanations)
		self.cdf_max = max(explanations)
		xs = np.linspace(self.cdf_min, self.cdf_max, 1000)
		cdfs = [0]
		from tqdm import tqdm
		for i in tqdm(range(1, len(xs))):
			cdfs.append(quad(density_kernel, xs[i-1], xs[i])[0])
		cdfs = np.cumsum(cdfs)
		cdfs = cdfs / cdfs[-1] * (1 - self.zero_pmf)
		self.cdf_spline = interp1d(xs, cdfs)

	def prob_measure(self, lo, hi):
		lo = lo * self.factor
		hi = hi * self.factor
		if lo <= self.cdf_min:
			cdf_lo = 0
		else:
			cdf_lo = self.cdf_spline(lo)
		if hi >= self.cdf_max:
			cdf_hi = 1 - self.zero_pmf
		else:
			cdf_hi = self.cdf_spline(hi)
		mu = cdf_hi - cdf_lo
		if self.zero_discrete and lo <= 0 <= hi:
			mu = mu + self.zero_pmf
		return mu

	def set_factor(self, val):
		self.factor = val

	def get_measure(self, b_val, exclude_val=None):
		m = sum(self.prob_measure(lo, hi) for lo, hi in b_val.lo_hi)
		if self.zero_discrete and b_val.contains(exclude_val) and exclude_val == 0:
			m = m - self.zero_pmf
		return m

class Model():
	def __init__(self, rule_union, data):
		self.rule_union = rule_union
		self.data = data

	def get_new_model_with_cur_params(self):
		rule_union = copy(self.rule_union)
		for r in rule_union.rules.values():
			for p in r.a_params:
				p.default_value = p.current_value
			for p in r.b_params:
				p.default_value = p.current_value
		return Model(rule_union, self.data)

	def pprint(self):
		s = io.StringIO()
		s.write(f'Composition: {self.rule_union.composition_structure}\n')
		for idx, rule in self.rule_union.rules.items():
			s.write(f'Rule {idx}: {rule.name}\n')
			s.write(f'  Applicability function parameters: \n')
			for p in rule.a_params:
				s.write(f'    {p.name}: {p.current_value}\n')
			s.write(f'  Behavior function parameters: \n')
			for p in rule.b_params:
				s.write(f'    {p.name}: {p.current_value}\n')
		return s.getvalue()

	def reset_rule_union(self):
		for r in self.rule_union.rules.values():
			for p in r.a_params:
				p.current_value = p.default_value
			for p in r.b_params:
				p.current_value = p.default_value

	def update_param_value(self, rule_idx, func_type, param_idx, new_val):
		self.rule_union.rules[rule_idx].get_params(func_type)[param_idx].current_value = new_val

	def shuffle_vis_order(self, typ):
		self.data.shuffle(typ)

	def get_metrics(self, rule_union, rule_idx=None):
		'''
		get metric values (cov, val, shp) for the rule union
		if rule_idx is not None, return additional metric values for the rule, taking into account
		'''
		if isinstance(rule_union, RuleUnionNull):
			assert rule_idx is None
			return 0, 0, 0
		coverages = []
		validities = []
		sharpnesses = []
		weights = []
		if rule_idx is not None:
			coverages_single = []
			validities_single = []
			sharpnesses_single = []
			rule = rule_union.get_rule_union_for_rule(rule_idx)
		for sent in self.data.sentences:
			for feu in sent.get_all_FEUs():
				a_val, b_val, rs = rule_union.get_a_b_func_value(feu)
				if not a_val:
					assert rs == tuple()
				coverages.append(a_val)
				validities.append(b_val.contains(feu.explanation))
				sharpnesses.append(1 - self.data.exp_measure.get_measure(b_val, feu.explanation))
				weights.append(1 / feu.L)
				if rule_idx is not None:
					a_val_single = rule_idx in rs
					coverages_single.append(a_val_single)
					_, b_val_single, _ = rule.get_a_b_func_value(feu)
					validities_single.append(b_val_single.contains(feu.explanation))
					sharpnesses_single.append(1 - self.data.exp_measure.get_measure(b_val_single, feu.explanation))
		coverages = np.array(coverages)
		validities = np.array(validities)
		sharpnesses = np.array(sharpnesses)
		weights = np.array(weights)
		cov = np.sum(coverages * weights) / np.sum(weights)
		val = np.sum(validities * weights * coverages) / np.sum(weights * coverages)
		shp = np.sum(sharpnesses * weights * coverages) / np.sum(weights * coverages)
		if rule_idx is None:
			return cov, val, shp
		else:
			coverages_single = np.array(coverages_single)
			validities_single = np.array(validities_single)
			sharpnesses_single = np.array(sharpnesses_single)
			cov_single = np.sum(coverages_single * weights) / np.sum(weights)
			val_single = np.sum(validities_single * weights * coverages_single) / np.sum(weights * coverages_single)
			shp_single = np.sum(sharpnesses_single * weights * coverages_single) / np.sum(weights * coverages_single)
			return cov, val, shp, cov_single, val_single, shp_single

	def get_metrics_whole_and_single(self, rule_idx=None):
		return self.get_metrics(self.rule_union, rule_idx=rule_idx)

	def get_metrics_without(self, rule_idx):
		return self.get_metrics(self.rule_union.get_rule_union_without_rule(rule_idx), rule_idx=None)

	def auto_tune(self, rule_idx, func_type, param_idx, start, stop, precision, method, metric, metric_for, min_metric_val):
		assert (func_type in ['a', 'b']) and (method in ['linear', 'binary'])
		assert (metric in ['cov', 'val', 'shp']) and (metric_for in ['whole', 'selected']) and (0 <= min_metric_val <= 1)
		metric_idx = {'cov': 0, 'val': 1, 'shp': 2}[metric]
		param = self.rule_union.get_rule_by_idx(rule_idx).get_params(func_type)[param_idx]
		old_val = param.current_value
		def is_feasible(v):
			param.current_value = v
			if metric_for == 'whole':
				return self.get_metrics_whole_and_single()[metric_idx] >= min_metric_val
			else:
				return self.get_metrics_whole_and_single(rule_idx)[metric_idx + 3] >= min_metric_val
		if method == 'linear':
			num_increments = int(np.floor(abs(stop - start) / precision) + 1)
			vals = np.linspace(start, stop, num_increments)
			for v in tqdm(vals):
				if is_feasible(v):
					return True, f'AutoTune successfully finds a solution with value of {v}. '
			param.current_value = old_val
			return False, 'AutoTune fails to find a solution. '
		else:
			if is_feasible(start):
				return True, f'AutoTune successfully finds a solution with value of {start}. '
			if not is_feasible(stop):
				param.current_value = old_val
				return False, f'AutoTune requires the stop value to be feasible in binary search, but it is not. '
			left = start  # always infeasible
			right = stop  # always feasible
			with tqdm(total=100) as pbar:
				while abs(right - left) > precision:
					mid = (left + right) / 2
					if is_feasible(mid):
						right = mid
					else:
						left = mid
					pbar.update(1)
				param.current_value = right
			return True, f'AutoTune successfully finds a solution with value of {right}. '

	def yield_sentence_visualizations(self, rule_idx=None, show='all'):
		'''
		yield a sequence of explanations.
		each explanation is a list of (word, color, applicable, preempted) for the given rule.
		applicable represents whether the applicability function returns True or False.
		if applicable is True, but the rule is ineffective due to lower precedence, then pre-empeted is True.
		only yield examples in which at least one EoE is applicable.
		'''
		assert show in ['all', 'at_least_one'] and show == 'all'  # at_least_one is not implemented yet
		rule = self.rule_union
		for idx in self.data.a_order_idxs:
			eoe = []
			sent = self.data.sentences[idx]
			label = sent.true_label
			pred = sent.prediction
			words = sent.words
			explanations = sent.explanations
			all_a = []
			for u in sent.get_all_FEUs():
				a, b, rs = rule.get_a_b_func_value(u)
				valid = b.contains(u.explanation)
				if rule_idx is None or rule_idx in rs:
					eoe.append((u.word, u.explanation, a, valid, rs))
				else:  # rule_idx not in rs
					eoe.append((u.word, u.explanation, False, valid, rs))

				all_a.append(a)
			if show == 'all' or any(all_a):
				yield idx, eoe, label, pred

	def yield_feu_visualization(self, rule_idx=None):
		'''
		yield a sequence of FEUs in their contexts.
		each explanation is a tuple of list-of-words, list-of-colors, true_label, prediction,
		the index of the FEU, ground truth explanation and predicted behavior range.
		The predicted behavior range is a list of (lo, hi)-intervals.
		'''
		rule = self.rule_union
		for idx in self.data.b_order_idxs:
			sent = self.data.sentences[idx]
			label = sent.true_label
			pred = sent.prediction
			words = sent.words
			explanations = sent.explanations
			applicable_feus = []
			for u in sent.get_all_FEUs():
				a, b, rs = rule.get_a_b_func_value(u)
				if a and (rule_idx is None or rule_idx in rs):
					applicable_feus.append((u.idx, u.explanation, b, rs))
			if len(applicable_feus) == 0:
				continue
			feu_idx, explanation, b, rs = random.choice(applicable_feus)
			yield idx, words, explanations, label, pred, feu_idx, explanation, b, rs

	def get_sentence_grouped_feu(self, idx):
		return self.data.sentences[idx]
