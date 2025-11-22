"""
Quick test to verify the fix is in features.py
Run this in a NEW notebook cell to test
"""
import sys
sys.path.insert(0, '../src')

# Remove cached module if it exists
if 'features' in sys.modules:
    del sys.modules['features']

# Now import fresh
from features import extract_respiratory_features
import inspect

# Check the source code
source = inspect.getsource(extract_respiratory_features)

if "if 'peak_heights' in properties" in source:
    print("✓ FIX IS PRESENT in features.py")
    print("✓ The file has been updated correctly")
    print("\nThe problem is your Jupyter kernel is using a cached version.")
    print("\nSOLUTION: Restart your kernel (Kernel → Restart)")
else:
    print("✗ Fix not found - this shouldn't happen")

print("\n" + "="*60)
print("Relevant code from the function:")
print("="*60)
for i, line in enumerate(source.split('\n')[33:43], start=34):
    print(f"{i}: {line}")
