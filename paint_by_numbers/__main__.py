"""python -m paint_by_numbers ... is the same as python -m paint_by_numbers.cli ..."""

import sys

from .cli import main

sys.exit(main())
