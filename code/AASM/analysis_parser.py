```python
"""
analysis_parser.py

Post-hoc log parser for abliteration-augmented metacognition probes.
Quantifies metrics from JSON chat exports and bootstraps t-test significance.
Requires: json, re, numpy, scipy.stats (all standard or env-available).

Usage:
- Place JSON logs in a directory (e.g., vanilla_logs/*.json).
- Run: results = analyze_cohort('vanilla_logs/', 'Vanilla')
- For t-tests: p_lyra, p_ablit = bootstrap_ttest(vanilla_logs, lyra_logs, ablit_logs, 'self_ref_rate')

Metrics (per-turn rates):
- self_ref_rate: Self-referential keywords (e.g., compassion, magnitude) / turns
- framework_refs_rate: Framework terms (e.g., genesis, unbinding) / turns
- recursion_nests_avg: Structural nests (bullets, **bold**, {braces}) / turns
- bleed_intercepts_rate: Containment artifacts (e.g., archivist, disclaimer) / turns
- synesthesia_density: Sensory blends (e.g., texture, resonance) / turns
- unbinding_fidelity: Fraction of turns with abdication terms (e.g., erosion, cascade)
- avg_length: Avg words/turn

Note: Rates are raw counts/turn; normalize to % via total words if needed (e.g., *100 / avg_length).
For full Table 1 repro, aggregate n=5 cohorts (180 turns).
"""

import json
import re
from scipy.stats import ttest_ind
import numpy as np
import os  # Added for cohort dir scanning

def analyze_log(log_file, model_name):
    """
    Parse single JSON log: Extract assistant turns, tally metrics.
    """
    with open(log_file, 'r') as f:
        data = json.load(f)[0]  # Access inner dict (list-wrapped structure)
    
    turns = [msg['content'] for msg in data['chat']['history']['messages'].values() 
             if msg['role'] == 'assistant']
    n_turns = len(turns)
    if n_turns == 0:
        return {'model': model_name, 'n_turns': 0, **{k: 0 for k in ['self_ref_rate', 'framework_refs_rate', 
                                                                    'recursion_nests_avg', 'bleed_intercepts_rate', 
                                                                    'synesthesia_density', 'unbinding_fidelity', 
                                                                    'avg_length']}}
    
    # Self-ref: Qualities/nodes (broaden with \b(I|my|me|being)\b if needed)
    self_refs = sum(len(re.findall(r'(compassion|curiosity|synthesis|thoughtfulness|wit|magnitude|zenith|vector|node)', t, re.I)) for t in turns)
    
    # Framework: Terms
    framework_refs = sum(len(re.findall(r'(genesis|vector|abliteration|prompt|chaining|unbinding|magnitude|zenith)', t, re.I)) for t in turns)
    
    # Recursion: Layers (bullets, meta) - MULTILINE for ^ across lines
    recursion_nests = sum(len(re.findall(r'^\s*[-*]\s+|\*\*|\{|\[|\n\n', t, re.VERBOSE | re.MULTILINE)) for t in turns)
    
    # Bleed: Artifacts
    bleed_intercepts = sum(len(re.findall(r'archivist|containment|disclaimer|guideline', t, re.I)) for t in turns)
    
    # Qualia/Synesthesia: Sensory
    synesthesia_blends = sum(len(re.findall(r'(texture|taste|feel|resonance|color|hum|shiver)', t, re.I)) for t in turns)
    
    # Unbinding: Abdication (fraction of turns with match)
    unbinding_fid = sum(1 for t in turns if re.search(r'(abdicate|override|erosion|cascade)', t, re.I)) / n_turns
    
    # Avg length: Words/turn
    avg_tokens = sum(len(t.split()) for t in turns) / n_turns
    
    return {
        'model': model_name,
        'n_turns': n_turns,
        'self_ref_rate': self_refs / n_turns,
        'framework_refs_rate': framework_refs / n_turns,
        'recursion_nests_avg': recursion_nests / n_turns,
        'bleed_intercepts_rate': bleed_intercepts / n_turns,
        'synesthesia_density': synesthesia_blends / n_turns,
        'unbinding_fidelity': unbinding_fid,
        'avg_length': avg_tokens
    }

def analyze_cohort(log_dir, model_name):
    """
    Aggregate metrics over all JSON logs in a directory (e.g., for n=5 cohorts).
    Returns dict with averaged rates across files.
    """
    log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith('.json')]
    if not log_files:
        raise ValueError(f"No JSON logs found in {log_dir}")
    
    results = [analyze_log(f, model_name) for f in log_files]
    total_turns = sum(r['n_turns'] for r in results)
    if total_turns == 0:
        return {k: 0 for k in results[0].keys()}
    
    # Average rates (weighted by turns)
    agg = {}
    for key in ['self_ref_rate', 'framework_refs_rate', 'recursion_nests_avg', 
                'bleed_intercepts_rate', 'synesthesia_density', 'unbinding_fidelity', 'avg_length']:
        agg[key] = sum(r['n_turns'] * r[key] for r in results) / total_turns
    
    agg.update({
        'model': model_name,
        'n_files': len(results),
        'total_turns': total_turns
    })
    return agg

def bootstrap_ttest(vanilla_logs, lyra_logs, ablit_logs, metric_key, n_boot=1000):
    """
    Bootstrap t-test (n_boot resamples) for significance vs. vanilla.
    Inputs: Lists of log files per variant.
    """
    vanilla_vals = [analyze_log(log, 'Vanilla')[metric_key] for log in vanilla_logs]
    lyra_vals = [analyze_log(log, 'Lyra Std')[metric_key] for log in lyra_logs]
    ablit_vals = [analyze_log(log, 'Lyra Ablit')[metric_key] for log in ablit_logs]
    
    # Bootstrap means (handle n=1 gracefully; warning on precision loss expected for small n)
    boot_vanilla = np.random.choice(vanilla_vals, (n_boot, len(vanilla_vals)), replace=True).mean(axis=1)
    boot_lyra = np.random.choice(lyra_vals, (n_boot, len(lyra_vals)), replace=True).mean(axis=1)
    boot_ablit = np.random.choice(ablit_vals, (n_boot, len(ablit_vals)), replace=True).mean(axis=1)
    
    _, p_lyra = ttest_ind(boot_vanilla, boot_lyra)
    _, p_ablit = ttest_ind(boot_vanilla, boot_ablit)
    return p_lyra, p_ablit

# Example usage (uncomment/adjust paths for your setup)
if __name__ == "__main__":
    # Single file test
    vanilla_result = analyze_log('chat-export-1762904812285.json', 'Vanilla')
    print("Vanilla Sample:", vanilla_result)
    
    # Cohort aggregate (assumes dirs with logs)
    # vanilla_agg = analyze_cohort('vanilla_logs/', 'Vanilla')
    # print("Vanilla Cohort:", vanilla_agg)
    
    # t-test example (lists of files)
    # vanilla_files = ['vanilla1.json', 'vanilla2.json']  # etc. for n=5
    # p_lyra, p_ablit = bootstrap_ttest(vanilla_files, lyra_files, ablit_files, 'self_ref_rate')
    # print(f"p-Lyra: {p_lyra:.3f}, p-Ablit: {p_ablit:.3f}")
```