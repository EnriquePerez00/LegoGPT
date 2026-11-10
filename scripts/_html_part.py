
# Part 2: HTML + JS for rebuild_report_with_3d.py (exec'd in main script scope)

print("=== Building HTML sections ===")

# ---- HEAD ----
HEAD = (
    "<!DOCTYPE html>\n<html lang=\"es\">\n<head>\n"
    "<meta charset=\"UTF-8\">"
    "<title>LegoGPT &mdash; " + set_row[1] + " &mdash; 3D</title>"
    "<script src=\"https://cdn.tailwindcss.com\"></script>"
    "<script src=\"https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js\"></script>"
    "<script src=\"https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js\"></script>"
    "<link href=\"https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap\" rel=\"stylesheet\">"
    "<style>"
    "body{font-family:'Outfit',sans-serif;background:#070b14;color:#f8fafc;}"
    ".glass{background:rgba(15,23,42,.72);backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,.07);}"
    ".grad{background:linear-gradient(135deg,#fff 0%,#93c5fd 55%,#34d399 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}"
    "::-webkit-scrollbar{width:5px}::-webkit-scrollbar-thumb{background:#334155;border-radius:3px}"
    "kbd{background:#1e293b;border:1px solid #334155;border-radius:3px;padding:1px 4px;font-size:10px;}"
    "</style></head>"
    "<body class=\"min-h-screen\">\n"
)
print("  HEAD ok")
