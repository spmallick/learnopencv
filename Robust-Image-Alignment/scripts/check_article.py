"""Execute the article's Python excerpts in their published order, headlessly."""

from pathlib import Path
import os
import re
import sys
import tempfile


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
text = (PROJECT / 'article.md').read_text()
blocks = re.findall(r'```python\n(.*?)\n```', text, re.S)
namespace = {'__name__': '__article_examples__'}
with tempfile.TemporaryDirectory(prefix='alignment-article-') as working:
    # Inputs resolve from their module; outputs must work in an unrelated directory.
    os.chdir(working)
    for index, block in enumerate(blocks, 1):
        exec(compile(block, f'article.md Python block {index}', 'exec'), namespace)
        print(f'Article Python block {index}: passed', flush=True)
print(f'VALIDATION PASSED: {len(blocks)} article Python excerpts')
