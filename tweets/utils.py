import re

def remove_all_non_text(doc):
    pattern = re.compile(r'@\S+|http\S+|pic.\S+|www\S+|\s*[^\w\d\s]\S*|\w*\d\w*')
    stripped = pattern.sub(''.test_str).strip()
    
    return stripped