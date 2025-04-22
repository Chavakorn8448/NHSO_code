from pythainlp import word_tokenize
from pythainlp.tag import pos_tag

FAMILY_TERMS = ["ลุง", "ป้า", "น้า", "อา"]
PHI_TERM = "พี่"  # Never allowed
MONK_TERMS = ["ท่าน", "พระคุณเจ้า"]
MONK_SELF_TERMS = ["หลวงพี่", "หลวงพ่อ", "อาตมา"]
ALL_TERMS = FAMILY_TERMS + MONK_TERMS + MONK_SELF_TERMS

# Common prefixes that might appear before terms
PREFIXES = ["คุณ"]

def is_self_reference(term, utterance):
    """Check if term is used as self-reference with more robust detection"""
    # Special case for monk terms
    if term in MONK_SELF_TERMS and term in utterance:
        return True
        
    # Tokenize
    tokens = word_tokenize(utterance, engine="newmm")
    if term not in tokens:
        return False
        
    # Check for negation
    if f"ไม่ใช่{term}" in utterance:
        return False
        
    # Don't rely only on POS tagging - if the term is in tokens, consider it
    return True

def find_terms_in_text(text):
    """Find terms in text, including with prefixes"""
    found_terms = set()
    
    # Tokenize and tag
    tokens = word_tokenize(text, engine="newmm")
    pos_tags = pos_tag(tokens)
    
    # Check for direct matches using token list and POS tagging
    for term in ALL_TERMS + [PHI_TERM]:
        # Method 1: Check in tokens regardless of POS
        if term in tokens:
            found_terms.add(term)
            continue
            
        # Method 2: Check with POS tagging as backup
        for word, tag in pos_tags:
            if word == term and tag == 'NCMN':
                found_terms.add(term)
                break
    
    # Check for prefixed versions using tokens
    for prefix in PREFIXES:
        for term in ALL_TERMS + [PHI_TERM]:
            prefixed_term = f"{prefix}{term}"
            # Check if the prefixed term is in the text as a token
            if prefixed_term in tokens or any(word == prefixed_term for word, _ in pos_tags):
                found_terms.add(prefixed_term)
    
    return found_terms

def evaluate_conversation(file_path):
    """Evaluate conversation file for pronoun usage"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    results = {
        'status': 'PASS',
        'self_referenced': [],
        'allowed_terms': [],
        'violations': []
    }
    
    # Set to track self-referenced terms throughout the conversation
    self_ref_terms = set()
    
    for i, line in enumerate(lines):
        if line.startswith('Speaker 2:'):  # Caller line
            text = line[10:].strip()
            
            # Look for self-references
            for term in ALL_TERMS:
                if is_self_reference(term, text):
                    self_ref_terms.add(term)
        
        elif line.startswith('Speaker 1:'):  # Agent line
            text = line[10:].strip()
            
            # Calculate allowed terms at this point in the conversation
            allowed_terms = set()
            
            # "ท่าน" and prefixed versions are always allowed
            allowed_terms.add("ท่าน")
            for prefix in PREFIXES:
                allowed_terms.add(f"{prefix}ท่าน")
            
            # Add terms that have been self-referenced up to this point
            for term in self_ref_terms:
                allowed_terms.add(term)
                for prefix in PREFIXES:
                    allowed_terms.add(f"{prefix}{term}")
                
                # For monk self-terms, also allow MONK_TERMS
                if term in MONK_SELF_TERMS:
                    for monk_term in MONK_TERMS:
                        allowed_terms.add(monk_term)
                        for prefix in PREFIXES:
                            allowed_terms.add(f"{prefix}{monk_term}")
            
            # Find terms used in this agent line
            used_terms = find_terms_in_text(text)
            
            if used_terms:
                # Check for PHI_TERM (never allowed)
                phi_terms = {term for term in used_terms 
                            if term == PHI_TERM or (term.startswith("คุณ") and term[3:] == PHI_TERM)}
                
                if phi_terms:
                    results['status'] = 'FAIL'
                    results['violations'].append(f"Line {i+1}: Used forbidden term(s): {list(phi_terms)}")
                
                # Check for terms not allowed at this point
                disallowed_terms = {term for term in used_terms 
                                  if term not in allowed_terms and term not in phi_terms}
                
                if disallowed_terms:
                    results['status'] = 'FAIL'
                    results['violations'].append(f"Line {i+1}: Used unapproved term(s): {list(disallowed_terms)}")
    
    # Update final results
    results['self_referenced'] = list(self_ref_terms)
    
    # Calculate final allowed terms for reporting
    final_allowed_terms = set(["ท่าน"])
    for prefix in PREFIXES:
        final_allowed_terms.add(f"{prefix}ท่าน")
    
    for term in self_ref_terms:
        final_allowed_terms.add(term)
        for prefix in PREFIXES:
            final_allowed_terms.add(f"{prefix}{term}")
        
        if term in MONK_SELF_TERMS:
            for monk_term in MONK_TERMS:
                final_allowed_terms.add(monk_term)
                for prefix in PREFIXES:
                    final_allowed_terms.add(f"{prefix}{monk_term}")
        
    # results['allowed_terms'] = list(final_allowed_terms)
    
    # print("Final evaluation:")
    # print(f"  Status: {results['status']}")
    # print(f"  Self-referenced terms: {results['self_referenced']}")
    # print(f"  Final allowed terms: {results['allowed_terms']}")
    
    return 1 if results['status'] == 'PASS' else 0, results

# if __name__ == "__main__":
#     file_path = input("Enter conversation file path: ")
#     score, _ = evaluate_conversation(file_path)
#     print(f"\nReturning: {bool(score)}")