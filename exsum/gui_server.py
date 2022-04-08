
import random, time, sys, pathlib, html, datetime, os, re
from copy import deepcopy as copy
from flask import Flask, render_template, url_for, request, redirect
import dill
dill.settings['recurse'] = True

import numpy as np

from exsum import Model

GREEN = '#2fa831'
RED = 'red'
BLUE = '#007fff'

def colorize(e):
    if e > 0:
        color = (int(e * 256), 0, 0)
    else:
        color = (0, int(-0.5 * e * 256), int(-e * 256))
    return f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'

def to_str(txt):
    if not isinstance(txt, float):
        return str(txt)
    if txt == 0:
        return '0'
    if abs(txt) <= 0.001:
        return f'{txt:0.2e}'
    else:
        return f'{txt:0.4f}'

def htmlize(word, underscore, bold, color, info=None):
    word = html.escape(word)
    if underscore:
        word = f'<u>{word}</u>'
    if bold:
        word = f'<b>{word}</b>'
    if info is None:
        tooltip = ''
    else:
        tooltip = f'data-toggle="tooltip" title="{info}" data-placement="top"'
    word = f'<span {tooltip} style="color: {color}; text-decoration-color: {color}">{word}</span>'
    return word

def now():
    return datetime.datetime.now().strftime('%d-%m-%y_%H-%M-%S')

def min_str(*args):
    i = np.argmin([float(e) for e in args])
    return args[i]

def max_str(*args):
    i = np.argmax([float(e) for e in args])
    return args[i]

def calc_bar(minn, maxx, width, exp_val):
    minn, maxx, exp_val = float(minn), float(maxx), float(exp_val)
    assert minn <= 0 and maxx >=0 and (minn <= exp_val <= maxx)
    full_range = maxx - minn
    zero_loc = (0 - minn) / full_range
    length = abs(exp_val) / full_range
    if exp_val <= 0:
        return (zero_loc - length) * width, length * width, BLUE
    else:
        return zero_loc * width, length * width, RED

def remove_tooltip(s):
    return re.sub('data-toggle="tooltip".*data-placement="top"', '', s)

class UIData():
    def __init__(self, model):
        self.model = model
        self.vis_mode = 'overall'
        self.vis_level = 'sentence'
        self.feu_filter = 'all'
        self.selected_rule = None

        self.whole_metrics = [-1, -1, -1]
        self.selected_metrics = [-1, -1, -1]
        self.compute_metrics()

        self.formula = self.model.rule_union.get_formula()
        self.cf_formula = None
        self.msg = None


    def select_rule(self, idx):
        self.selected_rule = idx
        if idx is not None:
            self.cf_formula = self.model.rule_union.get_rule_union_without_rule(idx).get_formula()
        else:
            self.vis_mode = 'overall'
        self.compute_metrics()

    def refresh(self):
        self.model.shuffle_vis_order('a')
        self.model.shuffle_vis_order('b')

    def update_param(self, func, idx, val):
        self.model.update_param_value(self.selected_rule, func, idx, val)
        self.compute_metrics()

    def compute_metrics(self):
        if self.selected_rule is None:
            self.whole_metrics = self.model.get_metrics_whole_and_single()
        else:
            cov, val, shp, cov_single, val_single, shp_single = self.model.get_metrics_whole_and_single(self.selected_rule)
            self.whole_metrics = (cov, val, shp)
            self.selected_metrics = (cov_single, val_single, shp_single)
            self.cf_metrics = self.model.get_metrics_without(self.selected_rule)

    def reset(self):
        self.select_rule(None)
        self.model.reset_rule_union()

    def save(self, fn, latest_fn=None):
        new_model = self.model.get_new_model_with_cur_params()
        with open(fn + '.txt', 'w') as f:
            f.write(new_model.pprint())
        with open(fn + '.pkl', 'wb') as f:
            dill.dump(new_model, f)
        if latest_fn is not None:
            with open(latest_fn + '.txt', 'w') as f:
                f.write(new_model.pprint())
            with open(latest_fn + '.pkl', 'wb') as f:
                dill.dump(new_model, f)
        self.msg = f'S Current model saved successfully as "{fn}.txt/pkl". '

    def auto_tune(self, args):
        rule_idx=self.selected_rule
        try:
            func_type = args['func_type']
            param_idx = int(args['param_idx'])
            start = float(args['start_val'])
            stop = float(args['stop_val'])
            precision = float(args['precision'])
            method = args['search_radio']
            metric = args['metric_radio']
            metric_for = args['for_radio']
            min_metric_val = float(args['target_val'])
        except:
            self.msg = 'E AutoTune Error: All fields need to be numeric. '
            return
        if start == stop:
            self.msg = 'E AutoTune Error: Start and stop values cannot be the same. '
            return
        param_range = self.model.rule_union.get_rule_by_idx(rule_idx).get_params(func_type)[param_idx].param_range
        lo, hi = param_range.get_lower_bound(), param_range.get_upper_bound()
        if not lo <= start <= hi:
            self.msg = f'E AutoTune Error: Start value is out of the parameter value range of {lo} to {hi}. '
            return
        if not lo <= stop <= hi:
            self.msg = f'E AutoTune Error: Stop value is out of the parameter value range of {lo} to {hi}. '
            return
        if precision <= 0:
            self.msg = 'E AutoTune Error: Precision must be strictly positive. '
            return
        if not 0 <= min_metric_val <= 1:
            self.msg = 'E AutoTune Error: Target metric value needs to be from 0 to 1. '
            return
        success, msg = self.model.auto_tune(rule_idx, func_type, param_idx,
            start, stop, precision, method, metric, metric_for, min_metric_val)
        if success:
            self.compute_metrics()
            self.msg = 'S ' + msg
        else:
            self.msg = 'E ' + msg

    def get_msg(self):
        msg = self.msg
        self.msg = None
        return msg

    def prepare_example_vis(self, ct_sent=20, ct_feu_1=12, ct_feu_2=30):
        # construct self.sentence_vis, self.feu_vis, both of which are pre-formatted html string
        ct = 0
        self.sentence_vis = []
        self.sent_data = []
        if self.vis_mode == 'overall':
            rule_idx = None
        else:
            rule_idx = self.selected_rule
        for v in self.model.yield_sentence_visualizations(rule_idx=rule_idx):
            idx, exsum, label, pred = v
            words, explanations, applicables, valids, ruless = map(list, zip(*exsum))
            colors = [colorize(e) for e in explanations]
            underlines = applicables
            bolds = [a and v for a, v in zip(applicables, valids)]
            words.insert(0, f'y={label} : {pred:0.2f}')
            colors.insert(0, GREEN if np.rint(pred) == label else RED)
            underlines.insert(0, False)
            bolds.insert(0, False)
            # infos = copy(explanations)
            infos = [f'{to_str(e)} R{",".join(map(str, rs))}' if len(rs) > 0 else to_str(e) for e, rs in zip(explanations, ruless)]
            infos.insert(0, None)
            html_lst = [htmlize(*args) for args in zip(words, underlines, bolds, colors, infos)]
            html_str = ' '.join(html_lst)
            self.sentence_vis.append(html_str)
            sent = self.model.get_sentence_grouped_feu(idx)
            data = [[w, [to_str(f) for f in fs], to_str(e)]
                    for w, fs, e in zip(html_lst[1:], sent.features, sent.explanations)]
            y_str = html_lst[0]
            self.sent_data.append((data, y_str))
            ct += 1
            if ct == ct_sent:
                break

        ct = 0
        self.feu_vis = []
        self.feu_exp_1 = []
        self.feu_data = []
        vis_iterator = self.model.yield_feu_visualization(rule_idx)
        for v in vis_iterator:
            idx, words, explanations, label, pred, feu_idx, true_e, pred_e, rs = v
            if self.feu_filter == 'invalid' and pred_e.contains(true_e):
                continue
            colors = [colorize(e) for e in explanations]
            words = [f'y={label} : {pred:0.2f}'] + words
            colors = [GREEN if np.rint(pred) == label else RED] + colors
            underlines = [False] * len(words)
            underlines[feu_idx + 1] = True
            bolds = [False] + [pred_e.contains(e) for e in explanations]
            infos = [to_str(e) for e in explanations]
            infos[feu_idx] = f'{to_str(true_e)} R{",".join(map(str, rs))}'
            infos.insert(0, None)
            html_lst = [htmlize(*args) for args in zip(words, underlines, bolds, colors, infos)]
            html_str = ' '.join(html_lst)
            self.feu_vis.append(html_str)
            self.feu_exp_1.append((true_e, pred_e, pred_e.contains(true_e)))
            sent = self.model.get_sentence_grouped_feu(idx)
            data = [[w.replace('<u>', '').replace('</u>', ''), [to_str(f) for f in fs], to_str(e)]
                    for w, fs, e in zip(html_lst[1:], sent.features, sent.explanations)]
            y_str = html_lst[0]
            self.feu_data.append((data, y_str, feu_idx))
            ct += 1
            if ct == ct_feu_1:
                break

        ct = 0
        self.feu_exp_2 = []
        for v in vis_iterator:
            idx, words, explanations, label, pred, feu_idx, true_e, pred_e, _ = v
            if self.feu_filter == 'invalid' and pred_e.contains(true_e):
                continue
            colors = [colorize(e) for e in explanations]
            words = [f'y={label} : {pred:0.2f}'] + words
            colors = [GREEN if np.rint(pred) == label else RED] + colors
            underlines = [False] * len(words)
            underlines[feu_idx + 1] = True
            bolds = [False] * len(words)
            html_lst = [htmlize(*args) for args in zip(words, underlines, bolds, colors)]
            self.feu_exp_2.append((true_e, pred_e, pred_e.contains(true_e)))
            sent = self.model.get_sentence_grouped_feu(idx)
            data = [[w.replace('<u>', '').replace('</u>', ''), [to_str(f) for f in fs], to_str(e)]
                    for w, fs, e in zip(html_lst[1:], sent.features, sent.explanations)]
            y_str = html_lst[0]
            self.feu_data.append((data, y_str, feu_idx))
            ct += 1
            if ct == ct_feu_2:
                break

class Server():
    def __init__(self, model, log_dir, save_dir):
        self.ui_data = UIData(model)
        self.app = Flask(__name__)
        self.app.jinja_env.globals.update(zip=zip, min_str=min_str, max_str=max_str, map=map, float=float,
                                          calc_bar=calc_bar, remove_tooltip=remove_tooltip)
        self.log = open(os.path.join(log_dir, f'log_{now()}.log'), 'w')
        self.save_dir = save_dir

        @self.app.route('/', methods=['POST', 'GET'])
        def index():
            self.log.write(f'{now()} homepage\n')
            self.log.flush()
            return self.render()

        @self.app.route('/select_rule/<int(signed=True):idx>')
        def select_rule(idx):
            self.log.write(f'{now()} select_rule {idx}\n')
            self.log.flush()
            self.ui_data.select_rule(idx)
            return self.render()

        @self.app.route('/overall_selected')
        def overall_selected():
            self.log.write(f'{now()} overall_selected\n')
            self.log.flush()
            if self.ui_data.vis_mode == 'overall':
                self.ui_data.vis_mode = 'selected'
            else:
                self.ui_data.vis_mode = 'overall'
            return self.render()

        @self.app.route('/sentence_feu')
        def sentence_feu():
            self.log.write(f'{now()} sentence_feu\n')
            self.log.flush()
            if self.ui_data.vis_level == 'sentence':
                self.ui_data.vis_level = 'feu'
            else:
                self.ui_data.vis_level = 'sentence'
            return self.render()

        @self.app.route('/feu_filter_switch')
        def feu_filter_switch():
            self.log.write(f'{now()} feu_filter_switch\n')
            self.log.flush()
            if self.ui_data.feu_filter == 'all':
                self.ui_data.feu_filter = 'invalid'
            else:
                self.ui_data.feu_filter = 'all'
            return self.render()

        @self.app.route('/save')
        def save():
            self.log.write(f'{now()} save\n')
            self.log.flush()
            fn = os.path.join(self.save_dir, f'save_{now()}')
            self.ui_data.save(fn, os.path.join(self.save_dir, 'latest'))
            return self.render()

        @self.app.route('/reset')
        def reset():
            self.log.write(f'{now()} reset\n')
            self.log.flush()
            self.ui_data.reset()
            return self.render()

        @self.app.route('/update_param/<string:func>/<int(signed=True):idx>/<string:val>')
        def update_param(func, idx, val):
            self.log.write(f'{now()} update_param {idx} {val}\n')
            self.log.flush()
            val = float(val)
            assert func in ['a', 'b']
            self.ui_data.update_param(func, idx, val)
            return self.render()

        @self.app.route('/refresh')
        def refresh():
            self.log.write(f'{now()} refresh\n')
            self.log.flush()
            self.ui_data.refresh()
            return self.render()

        @self.app.route('/auto_tune')
        def auto_tune():
            self.log.write(f'{now()} auto_tune ' + ' '.join([f'{k}:{v}' for k, v in request.args.items()]) + '\n')
            self.log.flush()
            self.ui_data.auto_tune(request.args)
            return self.render()

    def render(self):
        self.ui_data.prepare_example_vis()
        return render_template('index.html', ui_data=self.ui_data, now=time.time())

    def run(self, *args, **kwargs):
        self.app.run(*args, **kwargs)
