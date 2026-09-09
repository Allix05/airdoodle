"""The unified label space: digits, letters, and doodle categories, all
represented as 28x28 grayscale bitmaps (matching MNIST/EMNIST/Quick Draw's
shared numpy_bitmap format) so a single classifier can be trained across
all three sources.
"""

DIGITS = [str(d) for d in range(10)]
LETTERS = [chr(ord("A") + i) for i in range(26)]
DOODLES = ["cat", "house", "tree", "car", "sun", "star", "fish", "umbrella"]

CLASSES = DIGITS + LETTERS + DOODLES

DIGIT_OFFSET = 0
LETTER_OFFSET = len(DIGITS)
DOODLE_OFFSET = len(DIGITS) + len(LETTERS)

assert len(CLASSES) == 44
