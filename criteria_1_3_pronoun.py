from pythainlp import word_tokenize

FAMILY_TERMS = ["ลุง", "ป้า", "น้า", "อา"]
PHI_TERM = "พี่"  # Never
MONK_TERMS = ["ท่าน", "พระคุณเจ้า"]
MONK_SELF_TERMS = ["หลวงพี่", "หลวงพ่อ", "อาตมา"]
ALL_TERMS = FAMILY_TERMS + [PHI_TERM] + MONK_TERMS + MONK_SELF_TERMS

def is_self_reference(term, utterance):
    """Check if term is used as self-reference"""
    if term in MONK_SELF_TERMS and term in utterance:
        return True
        
    tokens = word_tokenize(utterance, engine="newmm")
    if term not in tokens:
        return False        

    if f"ไม่ใช่{term}" in utterance:
        return False
        
    return True

def evaluate_conversation(file_path):
    """Evaluate conversation file for pronoun usage"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    results = {
        'status': 'PASS',
        'self_referenced': None,
        'allowed_terms': [],
        'violations': []
    }
    
    valid_references = []
    
    for i, line in enumerate(lines):
        if line.startswith('Speaker 2:'):  # Caller
            text = line[10:].strip()
            for term in ALL_TERMS:
                if is_self_reference(term, text):
                    valid_references.append(term)
        
        elif line.startswith('Speaker 1:'):  # Agent
            text = line[10:].strip()
            used_terms = [t for t in word_tokenize(text, engine="newmm") if t in ALL_TERMS]
            
            if PHI_TERM in used_terms:
                results['status'] = 'FAIL'
                results['violations'].append(f"Line {i+1}: Used forbidden term: '{PHI_TERM}'")
            
            if valid_references:
                last_reference = valid_references[-1]
                results['self_referenced'] = last_reference
                
                if last_reference in FAMILY_TERMS:
                    results['allowed_terms'] = [last_reference]
                elif last_reference in MONK_SELF_TERMS:
                    results['allowed_terms'] = MONK_TERMS
                
                invalid_terms = [t for t in used_terms 
                               if t != PHI_TERM and t not in results['allowed_terms']]
                if invalid_terms:
                    results['status'] = 'FAIL'
                    results['violations'].append(f"Line {i+1}: Used unapproved terms: {invalid_terms}")
            else:
                if used_terms and any(t != PHI_TERM for t in used_terms):
                    results['status'] = 'FAIL'
                    results['violations'].append(f"Line {i+1}: Used terms before reference: {used_terms}")

    if results['status'] == 'FAIL':
        return 0, results
    else:
        return 1, results

if __name__ == "__main__":
    score, results = evaluate_conversation("/home/ckancha/rnd/tone_analysis/text_analysis/fail_fam/before.txt")
    print(score)
