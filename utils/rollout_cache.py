"""Trusted local source-rollout caches and offline candidate-change counts."""
from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from utils.ranking_cdr import DecisionSampler
from utils.executed_contact_switches import execution_trace, completed_contact_switches


def json_value(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=json_value, allow_nan=False)+'\n')


def best_point_changes(decisions, summary):
    """Raw CSO argmin changes and guided-marker changes, never RS triggers."""
    duration = summary.get('completion_time_s')
    if duration is None:
        duration = summary['steps']*summary.get('policy_interval_s', .02)
    complete = (len(decisions) == summary['steps'] and
                [d['step'] for d in decisions] == list(range(summary['steps'])))
    result = dict(best_contact_switch_events=None, best_contact_switch_frequency_hz=None,
        best_contact_change_definition='world-space CSO argmin candidate changes between consecutive ranking evaluations; first candidate excluded',
        best_contact_point_world_change_definition='world-space guided best contact changes between consecutive ranking evaluations; first candidate excluded',
        best_contact_trace_complete=complete)
    for prefix, local_field, world_fields in [
            ('best_contact', 'optimal_local', ('optimal_world',)),
            ('guided_contact', 'best_contact_local', ('best_contact_point_world', 'best_contact_world'))]:
        field = next((candidate for candidate in world_fields
                      if complete and all(d.get(candidate) is not None for d in decisions)),
                     local_field)
        if not complete or any(d.get(field) is None for d in decisions):
            result[prefix+'_switch_events'] = None
            result[prefix+'_switch_frequency_hz'] = None
            continue
        points = [np.asarray(d[field], dtype=float) for d in decisions]
        changes = sum(np.linalg.norm(b-a)>1e-9 for a,b in zip(points,points[1:]))
        result[prefix+'_switch_events'] = int(changes)
        result[prefix+'_switch_frequency_hz'] = changes/duration if duration else None
    result['best_contact_point_world_change_events'] = result['guided_contact_switch_events']
    result['best_contact_point_world_change_frequency_hz'] = result['guided_contact_switch_frequency_hz']
    return result


def save_source_cache(directory, bus, sampler, args):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if len(bus.summaries) != 1:
        raise ValueError('Cache one original trial per case process')
    summary = bus.summaries[0]
    trial = summary['trial']
    summary.update(rs_switch_events=sampler.switch_count, rs_evaluations=sampler.evaluations,
                   rs_candidate_events=len(sampler.events))
    summary.update(best_point_changes(sampler.decisions, summary))
    trace = execution_trace(bus.observations,bus.commands)
    physical = completed_contact_switches(trace)
    summary.update(mpc_contact_switches=physical['mpc_contact_switches'],
        mpc_contact_switch_events=physical['events'],
        mpc_contact_switch_definition=physical['definition'],
        mpc_contact_switch_parameters=physical['parameters'])
    for name, records in [(f'execution_contact_trace_trial{trial}',trace),
                          (f'ranking_decisions_trial{trial}',sampler.decisions)]:
        with (directory/(name+'.jsonl')).open('w') as f:
            for row in records: f.write(json.dumps(row,default=json_value,allow_nan=False)+'\n')
    dump_json(directory/'baseline_summary.json',summary)
    dump_json(directory/'source_config.json',vars(args))
    data = dict(version=1,args=vars(args),summaries=bus.summaries,
                observations=bus.observations,commands=bus.commands,
                decisions=sampler.decisions,events=sampler.events,
                switch_count=sampler.switch_count,evaluations=sampler.evaluations)
    path = directory/f'source_trial{trial}.pkl.gz'
    temp = path.with_suffix(path.suffix+'.tmp')
    with gzip.open(temp,'wb',compresslevel=3) as f: pickle.dump(data,f,protocol=4)
    temp.replace(path)
    print('[source-cache] '+str(path),flush=True)
    return path


def load_source_cache(path, args, trial):
    """Load only caches created locally by this workflow, never external pickle files."""
    path = Path(path)
    if path.is_dir(): path = path/f'source_trial{trial}.pkl.gz'
    with gzip.open(path,'rb') as f: data=pickle.load(f)
    if data['version']!=1 or len(data['summaries'])!=1 or data['summaries'][0]['trial']!=trial:
        raise ValueError('Incompatible source cache or seed')
    ignore = {'solver','ranking_metrics_jsonl','rollout_summary_json','rollout_trace_jsonl',
        'rollout_planner_snapshots','rollout_cache_dir','rollout_verbose','ranking_metric_gamma',
        'ranking_metric_eta','svg_screenshot_dir'}
    for key,value in data['args'].items():
        if key in ignore or key.startswith('rs_') or not hasattr(args,key): continue
        actual=getattr(args,key)
        if actual!=value: raise ValueError(f'Source cache configuration differs: {key}')
    sampler=DecisionSampler(best_change_distance=getattr(args,'rs_best_change_distance',.01))
    sampler.decisions=data['decisions'];sampler.events=data['events']
    sampler.switch_count=data['switch_count'];sampler.evaluations=data['evaluations']
    baseline=SimpleNamespace(summaries=data['summaries'],observations=data['observations'],
                             commands=data['commands'])
    return baseline,sampler
