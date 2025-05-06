import re

def chatml_to_sharegpt(
    chatml_text,
    include_system_messages=False,
    role_key_name="role",  # or "from"
    content_key_name="content" # or "value"
):
    """
    Converts a ChatML formatted string back into ShareGPT list format.
    Allows configuration for including system messages and output key names.
    """

    pattern = r"<\|(\w+)\|>(.*?)<\|end\|>"
    matches = re.findall(pattern, chatml_text, flags=re.DOTALL)
    
    sharegpt_conversation = []
    
    for role, content in matches:
        role_standardized = role.lower() 
        
        if role_standardized == "system" and not include_system_messages:
            continue  
        
        sharegpt_conversation.append({
            role_key_name: role_standardized, # Use the standardized role
            content_key_name: content.strip()
        })
    
    return sharegpt_conversation