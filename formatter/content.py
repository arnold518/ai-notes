import re

class Content:
    def __init__(self, lvl=0):
        self.lvl = lvl
            
    def str_latex(self):
        pass

    def str_beamer(self):
        return self.str_latex()
    
    def str_markdown(self):
        pass

    def process(self):
        pass

class ContentList(Content):
    def __init__(self, lvl=0):
        super().__init__(lvl)
        self.contents = []

    def add_content(self, content):
        self.contents.append(content)

    def str_markdown(self):
        ret = ""
        for content in self.contents:
            ret += content.str_markdown()
            if content != self.contents[-1]:
                ret += "\n"
        return ret

    def str_latex(self):
        ret = ""
        for content in self.contents:
            ret += content.str_latex()
            if content != self.contents[-1]:
                ret += "\n"
        return ret
    
    def str_beamer(self):
        ret = ""
        flag = False
        for content in self.contents:
            flag2 = self.lvl == 0 and not isinstance(content, (Admonition, Title, Section, Subsection))
            if not flag and flag2:
                ret += '\n\\begin{frame}[allowframebreaks]\n\n'
            elif flag and not flag2:                
                ret += '\\end{frame}\n\n'
            flag = flag2
            ret += content.str_beamer()
            if content != self.contents[-1]:
                ret += "\n"
        if flag:
            ret += '\\end{frame}\n'
        return ret

    def process(self):
        for content in self.contents:
            content.process()

class UnorderedList(Content):
    def __init__(self, lvl=0):
        super().__init__(lvl)
        self.items = []

    def add_item(self):
        self.items.append(ContentList(lvl=self.lvl+1))

    def str_markdown(self):
        ret = ""
        for item in self.items:
            t = item.str_markdown()
            t = ' ' * (self.lvl * 4) + '- ' + t.lstrip()
            if '\n' in t.strip() and item != self.items[-1]:
                t += '\n'
            ret += t
        return ret

    def str_latex(self):
        ret = ' ' * (self.lvl * 4) + "\\begin{itemize}\n"
        for item in self.items:
            t = item.str_latex()
            if '\n' in t.strip():
                ret += ' ' * (self.lvl * 4 + 4) + "\\item\n"
                ret += t
            else:
                ret += ' ' * (self.lvl * 4 + 4) + "\\item " + t.lstrip()
        ret += ' ' * (self.lvl * 4) + "\\end{itemize}\n"
        return ret
    
    def str_beamer(self):
        ret = ' ' * (self.lvl * 4) + "\\begin{itemize}\n"
        for item in self.items:
            t = item.str_beamer()
            if '\n' in t.strip():
                ret += ' ' * (self.lvl * 4 + 4) + "\\item\n"
                ret += t
            else:
                ret += ' ' * (self.lvl * 4 + 4) + "\\item " + t.lstrip()
        ret += ' ' * (self.lvl * 4) + "\\end{itemize}\n"
        return ret

    def process(self):
        for item in self.items:
            item.process()

class OrderedList(Content):
    def __init__(self, lvl=0):
        super().__init__(lvl)
        self.items = []

    def add_item(self):
        self.items.append(ContentList(lvl=self.lvl+1))

    def str_markdown(self):
        ret = ""
        for idx, item in enumerate(self.items):
            t = item.str_markdown()
            t = ' ' * (self.lvl * 4) + str(idx + 1) + '. ' + t.lstrip()
            if '\n' in t.strip() and item != self.items[-1]:
                t += '\n'
            ret += t
        return ret

    def str_latex(self):
        ret = ' ' * (self.lvl * 4) + "\\begin{enumerate}\n"
        for item in self.items:
            t = item.str_latex()
            if '\n' in t.strip():
                ret += ' ' * (self.lvl * 4 + 4) + "\\item\n"
                ret += t
            else:
                ret += ' ' * (self.lvl * 4 + 4) + "\\item " + t.lstrip()
        ret += ' ' * (self.lvl * 4) + "\\end{enumerate}\n"
        return ret

    def str_beamer(self):
        ret = ' ' * (self.lvl * 4) + "\\begin{enumerate}\n"
        for item in self.items:
            t = item.str_beamer()
            if '\n' in t.strip():
                ret += ' ' * (self.lvl * 4 + 4) + "\\item\n"
                ret += t
            else:
                ret += ' ' * (self.lvl * 4 + 4) + "\\item " + t.lstrip()
        ret += ' ' * (self.lvl * 4) + "\\end{enumerate}\n"
        return ret

    def process(self):
        for item in self.items:
            item.process()

class Title(Content):
    def __init__(self, title="", lvl=0):
        super().__init__(lvl)
        self.title = title

    def str_markdown(self):
        return '# ' + self.title

    def str_latex(self):
        return '\\chapter{' + self.title.split('. ', 1)[1].split('\n', 1)[0] + '}\n'
    
    def str_beamer(self):
        return '\\section{' + self.title.split('\n', 1)[0] + '}\n'

class Section(Content):
    def __init__(self, title="", lvl=0):
        super().__init__(lvl)
        self.title = title

    def str_markdown(self):
        return '## ' + self.title

    def str_latex(self):
        tex = self.title.split('\n', 1)[0]
        tex = re.sub(r'(?<!\\)#', r'\\#', tex)
        tex = re.sub(r'(?<!\\)&', r'\\&', tex)
        return '\\section{' + tex + '}\n'
    
    def str_beamer(self):
        tex = self.title.split('\n', 1)[0]
        tex = re.sub(r'(?<!\\)#', r'\\#', tex)
        tex = re.sub(r'(?<!\\)&', r'\\&', tex)
        return '\\subsection{' + tex + '}\n'

class Subsection(Content):
    def __init__(self, title="", lvl=0):
        super().__init__(lvl)
        self.title = title

    def str_markdown(self):
        return '### ' + self.title

    def str_latex(self):
        tex = self.title.split('\n', 1)[0]
        tex = re.sub(r'(?<!\\)#', r'\\#', tex)
        tex = re.sub(r'(?<!\\)&', r'\\&', tex)
        return '\\subsection{' + tex + '}\n'
    
    def str_beamer(self):
        tex = self.title.split('\n', 1)[0]
        tex = re.sub(r'(?<!\\)#', r'\\#', tex)
        tex = re.sub(r'(?<!\\)&', r'\\&', tex)
        return '\\subsubsection{' + tex + '}\n'

def convert_links(text, type):
    """
    Detects '**Definition a.b**', '**Concept a.b**', '**Theorem a.b**', '**Example a.b**'
    (where a, b are numbers) in a string and replaces them with:
    '**[Definition a.b](./a.md#definition-a-b)**' without additional bolding on the link.
    """
    # Regex pattern to match "**Definition a.b**", "**Concept a.b**", etc.
    pattern = r'\*\*(Definition|Concept|Theorem|Example|definition|concept|theorem|example) (\d+)\.(\d+)\*\*'

    # Replace with Markdown-style link (without extra bolding on the link)
    def replace_match(match):
        term, a, b = match.groups()
        if type == "markdown" : return f'[{term.capitalize()} {a}.{b}](./{a}.md#{term.lower()}-{a}-{b})'
        if type == "latex" : return f'\\hyperref[{term.lower()}:{a}.{b}]{{{term.capitalize()} {a}.{b}}}'
    
    return re.sub(pattern, replace_match, text)

class Text(Content):
    def __init__(self, text="", lvl=0):
        super().__init__(lvl)
        self.text = text

    def append(self, text):
        self.text += text

    def str_markdown(self):
        ret = ""
        for line in self.text.split('\n'):
            if line == "": continue
            md = convert_links(line, "markdown")
            ret += ' ' * (self.lvl * 4) + md + '\n'
        return ret

    def str_latex(self):
        ret = ""
        for line in self.text.split('\n'):
            if line == "": continue
            tex = line
            tex = tex.replace('  ', '\\\\')
            tex = re.sub(r'(?<!\\)#', r'\\#', tex)
            tex = re.sub(r'(?<!\\)&', r'\\&', tex)
            tex = re.sub(r'`([^`]+)`', r'\\verb|\1|', tex)
            tex = convert_links(tex, "latex")
            tex = re.sub(r'(?<!\w)[_*]{2}(.*?)[_*]{2}(?!\w)', r'\\textbf{\1}', tex)

            markdown_link_pattern = re.compile(r'\[(.*?)\]\((.*?)\)')
            tex = markdown_link_pattern.sub(r'\\href{\2}{\1}', tex)
            ret += ' ' * (self.lvl * 4) + tex + '\n'
        return ret
        

class Equation(Content):
    def __init__(self, equation="", lvl=0):
        super().__init__(lvl)
        self.equation = equation

    def append(self, text):
        self.equation += text

    def str_markdown(self):
        ret = ' ' * (self.lvl * 4) + '$$\n'
        for line in self.equation.split('\n'):
            if line == "": continue
            ret += ' ' * (self.lvl * 4) + line + '\n'
        ret += ' ' * (self.lvl * 4) + '$$\n'
        return ret

    def str_latex(self):
        ret = ' ' * (self.lvl * 4) + '$$\n'
        for line in self.equation.split('\n'):
            if line == "": continue
            ret += ' ' * (self.lvl * 4) + line + '\n'
        ret += ' ' * (self.lvl * 4) + '$$\n'
        return ret

class Image(Content):
    def __init__(self, image_path="", scale=100, lvl=0):
        super().__init__(lvl)
        self.image_path = '../' + image_path
        self.scale = scale

    def str_markdown(self):
        ret = ' ' * (self.lvl * 4) + '<center>\n'
        ret += ' ' * (self.lvl * 4) + '![](' + self.image_path + '){: width="' + str(self.scale) + '%"}\n'
        ret += ' ' * (self.lvl * 4) + '</center>\n'
        return ret

    def str_latex(self):
        ret = ' ' * (self.lvl * 4) + '\\begin{figure}[H]\n'
        ret += ' ' * (self.lvl * 4 + 4) + '\\centering\n'
        ret += ' ' * (self.lvl * 4 + 4) + '\\includegraphics[width=' + str(self.scale/100) + '\\textwidth]{' + self.image_path + '}\n'
        ret += ' ' * (self.lvl * 4) + '\\end{figure}\n'
        return ret

class CodeBlock(Content):
    def __init__(self, code="", lvl=0):
        super().__init__(lvl)
        self.code = code

    def append(self, text):
        self.code += text

    def str_markdown(self):
        ret = ' ' * (self.lvl * 4) + '```\n'
        for line in self.code.split('\n'):
            if line == "": continue
            ret += ' ' * (self.lvl * 4) + line + '\n'
        ret += ' ' * (self.lvl * 4) + '```\n'
        return ret

    def str_latex(self):
        ret = ' ' * (self.lvl * 4) + '\\begin{verbatim}\n'
        for line in self.code.split('\n'):
            if line == "": continue
            ret += ' ' * (self.lvl * 4) + line + '\n'
        ret += ' ' * (self.lvl * 4) + '\\end{verbatim}\n'
        return ret

class Separator(Content):
    def __init__(self, lvl=0):
        super().__init__(lvl)

    def str_markdown(self):
        return ' ' * (self.lvl * 4) + '---\n'

    def str_latex(self):
        return ' ' * (self.lvl * 4) + '\\par\\noindent\\textcolor{gray}{\\hdashrule{\\textwidth}{0.4pt}{1pt 2pt}}\n'

class Admonition(Content):
    def __init__(self, chp=0, num=0, type="", title="", lvl=0):
        super().__init__(lvl)
        self.chp = chp
        self.num = num
        self.type = type
        self.title = title
        self.contentlist = ContentList(lvl+1)

    def str_markdown(self):
        ret = ' ' * (self.lvl * 4)
        ret += f'!!! {self.type.lower()} "{self.type} {self.chp}.{self.num} <a id=\"{self.type.lower()}-{self.chp}-{self.num}\"></a>: {self.title}"\n'
        ret += self.contentlist.str_markdown()
        return ret

    def str_latex(self):
        ret = ' ' * (self.lvl * 4)
        ret += f'\\begin{{{self.type.lower()}}}[{self.chp}.{self.num}][{self.title}]\n'
        ret += self.contentlist.str_latex()
        ret += f'\\end{{{self.type.lower()}}}\n'  # Changed to self.type.lower()
        return ret

    def str_beamer(self):
        ret = ""
        if self.lvl == 0:
            ret += f'\\begin{{frame}}[allowframebreaks]\n\n'
        ret += ' ' * (self.lvl * 4)
        ret += f'\\begin{{my{self.type.lower()}block}}{{{self.chp}.{self.num}}}{{{self.title}}}\n'
        ret += self.contentlist.str_latex()
        ret += f'\\end{{my{self.type.lower()}block}}\n'
        if self.lvl == 0:            
            ret += f'\n\\end{{frame}}\n'
        return ret

    def process(self):
        self.contentlist.process()

class Proof(Content):
    def __init__(self, lvl=0):
        super().__init__(lvl)
        self.contentlist = ContentList(lvl + 1)

    def str_markdown(self):
        ret = ' ' * (self.lvl * 4) + '!!! proof\n'
        ret += self.contentlist.str_markdown()
        return ret

    def str_latex(self):
        ret = ' ' * (self.lvl * 4) + '\\begin{proof}\n'
        ret += self.contentlist.str_latex()
        ret += ' ' * (self.lvl * 4) + '\\end{proof}\n'
        return ret

    def process(self):
        self.contentlist.process()
