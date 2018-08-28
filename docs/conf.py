import os
import sys
import datetime
import sphinx_readable_theme

sys.path.insert(0, os.path.abspath("../"))
import cartesian


project = "Cartesian"


copyright = "{}, Markus Quade".format(datetime.datetime.now().year)
author = "Markus Quade"
version = release = cartesian.__version__

master_doc = "index"

extensions = [
    "sphinxcontrib.apidoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]


html_sidebars = {
    'index': [
        'sidebarintro.html',
    ],
    '**': [
        'sidebarintro.html',
        'localtoc.html',
        'relations.html',
        'searchbox.html',
    ]
}

apidoc_module_dir = "../cartesian"
apidoc_excluded_paths = ["tests"]


autodoc_default_flags = ["members"]
autodoc_member_order = "bysource"
autoclass_content = "init"

language = None

templates_path = ['_templates']
exclude_patterns = ["_build"]
#pygments_style = "sphinx"

add_module_names = True
add_function_parentheses = False
todo_include_todos = True

html_theme_path = [sphinx_readable_theme.get_html_theme_path()]
html_theme = "readable"
html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = True

default_role = "any"


# for sidebarintro.html
html_context = {
    "github_user": "ohjeah",
    "github_repo": project.lower(),
    "github_button": True,
    "github_banner": True,
    "github_type": "star",
    "github_count": True,
    "badge_branch": "master",
    "pypi_project": project,
}


