import os
import sys
import datetime
sys.path.insert(0, os.path.abspath('../../'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
]
templates_path = ['_templates']
autodoc_default_flags = ['members']
autodoc_member_order = 'bysource'
autoclass_content = 'init'
source_suffix = ['.rst']

master_doc = 'index'
project = 'Cartesian'
copyright = '{}, Markus Quade'.format(datetime.datetime.now().year)
author = 'Markus Quade'

import cartesian
version = release = cartesian.__version__

language = None
exclude_patterns = ['_build']

pygments_style = 'sphinx'
add_module_names = True
add_function_parentheses = False
todo_include_todos = True

html_theme = 'alabaster'
html_theme_options = {
    'show_powered_by': False,
    'github_user': 'Ohjeah',
    'github_repo': 'cartesian',
    'github_banner': True,
    'github_type': 'star',
    'show_related': False,
    'description': "",
}

html_sidebars = {
    'index': [
        'about.html',
        'sidebarintro.html',
        'hacks.html',  # kudos to kenneth reitz
    ],
    '**': [
        'about.html',
        'localtoc.html',
        'hacks.html',
    ]
}

htmlhelp_basename = 'cartesiandoc'

html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = True

default_role = 'any'

import recommonmark
from recommonmark.parser import CommonMarkParser
from recommonmark.transform import AutoStructify

source_parsers = {'.md': CommonMarkParser}

# app setup hook
github_doc_root = 'https://github.com/Ohjeah/cartesian/tree/master/docs/'


def setup(app):
    app.add_config_value(
        'recommonmark_config', {
            'url_resolver': lambda url: github_doc_root + url,
            'auto_toc_tree_section': 'Contents',
            'enable_eval_rst': True,
            'enable_auto_doc_ref': True,
        }, True)
    app.add_transform(AutoStructify)
