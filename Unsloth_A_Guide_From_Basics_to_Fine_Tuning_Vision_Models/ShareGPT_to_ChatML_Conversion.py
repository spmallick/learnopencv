def sharegpt_to_chatml(sharegpt_conversation, system_prompt="You are a helpful assistant.", add_default_system_prompt_if_missing=True):
    """
    Converts a ShareGPT style conversation (list of dicts) into a ChatML string.
    Handles common ShareGPT role keys ('from', 'role') and content keys ('value', 'content').
    Handles common ShareGPT roles ('human', 'user', 'gpt', 'assistant', 'system').
    """
    chatml_parts = []
    has_system_prompt_in_data = False

    for turn in sharegpt_conversation:
        role_key = 'role' if 'role' in turn else 'from'
        if turn.get(role_key) == "system":
            has_system_prompt_in_data = True
            break
            
    if add_default_system_prompt_if_missing and not has_system_prompt_in_data and system_prompt:
        chatml_parts.append(f"<|system|>{system_prompt.strip()}<|end|>")
    
    for turn in sharegpt_conversation:
        role_key = 'role' if 'role' in turn else 'from'
        content_key = 'content' if 'content' in turn else 'value'

        if role_key not in turn or content_key not in turn:
            print(f"Skipping turn due to missing keys: {turn}") 
            continue

        role = turn[role_key]
        content = turn[content_key].strip()
        
        if role in ["user", "human"]:
            chatml_parts.append(f"<|user|>{content}<|end|>")
        elif role in ["assistant", "gpt", "model"]:
            chatml_parts.append(f"<|assistant|>{content}<|end|>")
        elif role == "system":
            chatml_parts.append(f"<|system|>{content}<|end|>")
        else:
            raise ValueError(f"Unknown role: {role} in turn: {turn}")
            
    return "\n".join(chatml_parts)