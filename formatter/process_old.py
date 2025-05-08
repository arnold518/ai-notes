# Import necessary classes and functions
from content import *
from markdown_parser import MarkdownParser

def process(n):
    path = '../docs/MFDNN/'
    parser = MarkdownParser(path + str(n) + '.md')
    contentlist = parser.parse()
    with open(path + 'formatted/' + str(n) + '.md', 'w') as file:
        file.write(contentlist.str_markdown())
    
    contentlist.process()
    
    with open(path + 'formatted/' + str(n) + '.tex', 'w') as file:
        file.write(contentlist.str_latex())

if __name__ == "__main__":
    # process(1)
    for i in range(1, 14):
        process(i)
