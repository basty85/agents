import sys
import os

print("Python executable:", sys.executable)
print("sys.path:")
for p in sys.path:
    print("  ", p)
print(".venv detected:", os.path.basename(sys.executable))

words = ['Hello', 'world', 'Python', 'is', 'great']
sentence = " ".join(words)
print(sentence)  # Output: "Hello world Python is great"