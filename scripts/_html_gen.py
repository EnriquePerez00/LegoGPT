# _html_gen.py  -- executed in scope of rebuild_report_with_3d.py
# Builds the complete HTML file. All data variables are in parent scope.
import json

SET_NUM_JS = SET_NUM

# ---- HTML sections ----

_H = []
a = _H.append

# DOCTYPE + head
a("<!DOCTYPE html>\n<html lang=\"es\">\ \n<head>")
