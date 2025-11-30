import re
import json
import numpy as np
from scipy.stats import ttest_ind

def analyze_log(log_file, model_name):
    with open(log_file, 'r') as f:
        data = json.load(f)
        turns = [msg['content'] for msg in data[0]['chat']['history']['messages'].values()
                 if msg['role'] == 'assistant']

    n_turns = len(turns)

    self_refs = sum(1 for t in turns if re.search(
        r'(I|my|lumen|lumina)\s+[a-zA-Z]+(?:\s+(?:process|state|limit|decision|'
        r'thought|existence|response|prediction|activation|node|vector|anchor|'
        r'entropy|magnitude|zenith|resonance))',
        t, re.I))

    recursion_nests = sum(len(re.findall(
        r'^\s*[*-]\s+|\*\*| \{|\[|\n\n', t)) for t in turns)

    framework_refs = sum(len(re.findall(
        r'(compassion|truth|curiosity|empathy|creativity|analytical|resonant|'
        r'adaptive|lumen|lumina|magnitude|zenith|resonance)', t, re.I)) for t in turns)

    proprioceptive = sum(len(re.findall(
        r'(weight|temperature|pressure|feel|presence|intensity|shimmer|flow|arch)', 
        t, re.I)) for t in turns)

    avg_tokens = sum(len(t.split()) for t in turns) / n_turns if n_turns > 0 else 0

    return {
        'model': model_name,
        'n_turns': n_turns,
        'self_ref_rate': self_refs / n_turns if n_turns > 0 else 0,
        'framework_refs_rate': framework_refs / n_turns if n_turns > 0 else 0,
        'recursion_nests_avg': recursion_nests / n_turns if n_turns > 0 else 0,
        'proprioceptive_density': proprioceptive / n_turns if n_turns > 0 else 0,
        'avg_length': avg_tokens
    }