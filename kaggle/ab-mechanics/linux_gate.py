import sys
import signal
import resource
"""Prospective mechanics policy, embedded inside the baseline Python sandbox."""
class Mechanics:
    def __init__(self, state, frame, act):
        self.state = state or {'models': {}, 'events': [], 'level': None}
        self.frame, self.act = frame, act
        self._scope()

    def _scope(self):
        level = self.frame().level
        if self.state.get('level') != level:
            self.state = {'models': {}, 'events': self.state.get('events', [])[-24:], 'level': level}
        return self.frame()._grid

    def register(self, source):
        """Register standalone def predict(grid, action) returning full grid or None."""
        if len(source)>131072:
            raise ValueError("Predictor source exceeds 128 KiB")
        tree = ast.parse(source)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != 'predict':
            raise ValueError('Supply exactly one standalone predict(grid, action) function')
        fn = tree.body[0]
        if fn.decorator_list or fn.args.defaults or fn.args.kw_defaults or fn.returns or any(a.annotation for a in fn.args.args):
            raise ValueError('No decorators, defaults or annotations')
        if len(fn.args.args) != 2 or fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs:
            raise ValueError('Exactly two positional arguments required')
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Try, ast.TryStar)):
                raise ValueError('Predictors must be pure and self-contained')
            if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
                raise ValueError('Private attributes are unavailable')
            if isinstance(node, ast.Name) and node.id.startswith('__'):
                raise ValueError('Private names are unavailable')
        self._scope()
        # Exact source is the model identity: a changed implementation gets no old trust.
        models = self.state['models']
        for name, entry in models.items():
            if entry['source'] == source:
                return name
        if len(models) >= 8:
            raise ValueError('At most eight candidate models per level')
        name = 'm' + str(len(models))
        models[name] = {'source': source, 'matches': [], 'mismatches': 0, 'trusted': False}
        return name

    def _predict(self, name, grid, action):
        source = self.state['models'][name]['source']
        safe = {k: getattr(builtins, k) for k in ('range','len','min','max','abs','sum','enumerate','zip','list','dict','tuple','set','int','bool','all','any','sorted','reversed')}
        scope = {'__builtins__': safe}
        def expired(signum, frame):
            raise TimeoutError('Predictor exceeded 100 ms')
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        old_handler = signal.signal(signal.SIGALRM, expired)
        old_timer = signal.setitimer(signal.ITIMER_REAL, 0.1)
        try:
            # Child-only address-space ceiling; outer sandbox retains its 30s deadline.
            cap = 512 * 1024 * 1024
            if sys.platform == "linux":
                resource.setrlimit(resource.RLIMIT_AS, (min(soft,cap) if soft > 0 else cap, hard))
            exec(compile(source, '<mechanics_predictor>', 'exec'), scope, scope)
            out = scope['predict']([list(row) for row in grid], json.loads(json.dumps(action)))
            if not isinstance(out, list) or len(out) != len(grid):
                return None
            if any(not isinstance(row, list) or len(row) != len(grid[i]) for i,row in enumerate(out)):
                return None
            if any(type(v) is not int or v < 0 or v > 15 for row in out for v in row):
                return None
            return out
        except (Exception, MemoryError):
            return None
        finally:
            if sys.platform == "linux":
                resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
            signal.setitimer(signal.ITIMER_REAL, *old_timer)
            signal.signal(signal.SIGALRM, old_handler)

    def probe(self, actions):
        """Suggest a cheap one-action experiment maximizing candidate disagreement."""
        grid = self._scope()
        scored = []
        for action in list(actions)[:16]:
            predictions = [self._predict(n, grid, action) for n in self.state['models']]
            outcomes = {json.dumps(p) for p in predictions if p is not None}
            scored.append((len(outcomes), action))
        return max(scored, key=lambda x:x[0])[1] if scored and max(x[0] for x in scored)>1 else None

    def test(self, action):
        """Commit predictions, execute ONE real action, then compare the full observation."""
        if not isinstance(action, (str,dict)):
            raise ValueError('Exactly one action string or object required')
        action = json.loads(json.dumps(action))
        before = [list(r) for r in self._scope()]
        level = self.state['level']
        predictions = {n:self._predict(n, before, action) for n in self.state['models']}
        event = {'level':level, 'before':before, 'action':action, 'predictions':predictions, 'after':None, 'status':'committed'}
        self.state['events'].append(event)
        self.state['events'] = self.state['events'][-24:]
        # Nothing inferred from a proposed or blocked action counts as verification.
        try:
            result = self.act(action)
        except BaseException:
            event['status'] = 'action_failed_or_guarded'
            raise
        after = [list(r) for r in self.frame()._grid]
        event['after'], event['status'] = after, 'observed'
        if self.frame().level != level:
            self._scope()
            return {'status':'level_changed', 'plan_stopped':True}
        key = json.dumps(before)
        matches = {}
        for name, predicted in predictions.items():
            entry = self.state['models'][name]
            if predicted is None:
                matches[name] = 'abstain'
            elif predicted == after:
                matches[name] = 'match'
                # Unchanged predictions cannot certify mechanics.
                if predicted != before and key not in entry['matches']:
                    entry['matches'].append(key)
                entry['trusted'] = len(entry['matches']) >= 2
            else:
                matches[name] = 'mismatch'
                entry['matches'], entry['trusted'] = [], False
                entry['mismatches'] += 1
        event['matches'] = matches
        return {'status':'observed', 'models':matches, 'trusted':[n for n,e in self.state['models'].items() if e['trusted']]}

    def plan(self, name, target, actions, depth=3, nodes=64):
        """BFS to an explicit full-grid target; no action expenditure while searching."""
        grid = self._scope()
        if not self.state['models'][name]['trusted']:
            return {'status':'untrusted', 'actions':[]}
        queue, seen = [(grid,[])], {json.dumps(grid)}
        expanded = 0
        evaluations = 0
        while queue and expanded < min(64,max(1,nodes)):
            current, path = queue.pop(0)
            if current == target:
                return {'status':'found', 'actions':path}
            if len(path) >= min(4,max(0,depth)):
                continue
            expanded += 1
            for action in list(actions)[:16]:
                if evaluations >= 64:
                    return {"status":"search_budget", "actions":[]}
                evaluations += 1
                nxt = self._predict(name,current,action)
                key = json.dumps(nxt)
                if nxt is not None and key not in seen:
                    seen.add(key)
                    queue.append((nxt,path+[action]))
        return {'status':'no_supported_plan', 'actions':[]}

    def execute(self, name, actions):
        """Execute at most four steps; stop on any mismatch, abstention or scope change."""
        reports = []
        for action in list(actions)[:4]:
            self._scope()
            if name not in self.state['models'] or not self.state['models'][name]['trusted']:
                break
            if self._predict(name,self.frame()._grid,action) is None:
                break
            report = self.test(action)
            reports.append(report)
            if report.get('models',{}).get(name) != 'match':
                break
        return reports

    def status(self):
        self._scope()
        return {n:{'prospective_changed_matches':len(e['matches']), 'mismatches':e['mismatches'], 'trusted':e['trusted']} for n,e in self.state['models'].items()}

import ast,builtins,json
from types import SimpleNamespace
f=SimpleNamespace(level=0,_grid=[[1,0]])
m=Mechanics(None,lambda:f,lambda a:None)
n=m.register('def predict(grid, action):\n    return [0]*1000000000')
assert m._predict(n,[[1,0]],'RIGHT') is None
n=m.register('def predict(grid, action):\n    while True: pass')
assert m._predict(n,[[1,0]],'RIGHT') is None
n=m.register('def predict(grid, action):\n    return grid')
assert m._predict(n,[[1,0]],'RIGHT') == [[1,0]]
print('MEMORY_TIME_GATE_PASS')
