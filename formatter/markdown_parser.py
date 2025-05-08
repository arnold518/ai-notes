from content import *

class MarkdownParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.content = self.read_file()

    def read_file(self):
        with open(self.file_path, 'r') as file:
            return file.readlines()
        
    def parse(self):
        contentlist = ContentList()
        ST = [contentlist]

        processed_content = []
        for line in self.content:
            sline = line.lstrip(' ')
            if sline == "<center>\n" or sline == "</center>\n" or sline == "<center>" or sline == "</center>": continue
            if sline.startswith('- '):
                processed_content.append(' ' * (len(line) - len(sline)) + '- \n')
                processed_content.append(' ' * (len(line) - len(sline) + 4) + sline[2:])
                processed_content.append('\n')
            elif bool(re.match(r'^\d+\. ', sline)):
                t = sline.split('. ', 1)
                processed_content.append(' ' * (len(line) - len(sline)) + t[0] + '. \n')
                processed_content.append(' ' * (len(line) - len(sline) + 4) + t[1])
                processed_content.append('\n')
            elif sline != '$$\n' and sline.startswith('$$') and sline.endswith('$$\n'):
                processed_content.append(' ' * (len(line) - len(sline)) + '$$\n')
                processed_content.append(' ' * (len(line) - len(sline)) + sline[2:-3] + '\n')
                processed_content.append(' ' * (len(line) - len(sline)) + '$$\n')
            else :
                processed_content.append(line)
        self.content = processed_content

        for lineno, line in enumerate(self.content):
            indent_level = (len(line) - len(line.lstrip(' '))) // 4
            sline = line.lstrip(' ')
            
            print(f"{lineno}:{indent_level}:{sline}", end="")

            if isinstance(ST[-1], Equation):
                if sline == "\n":
                    ST[-1].append(sline)
                else:
                    if sline.startswith('$$'):
                        ST.pop()
                    else:
                        ST[-1].append((len(line) - len(sline) - ST[-1].lvl * 4) * ' ' + sline)
            elif isinstance(ST[-1], CodeBlock):
                if sline == "\n":
                    ST[-1].append(sline)
                else:
                    if sline.startswith('```'):
                        ST.pop()
                    else:
                        ST[-1].append((len(line) - len(sline) - ST[-1].lvl * 4) * ' ' + sline)
            else:
                if sline.startswith('# '):
                    while len(ST) > 1: ST.pop()
                    title = Title(sline[2:], lvl=indent_level)
                    ST[-1].add_content(title)
                elif sline.startswith('## '):
                    while len(ST) > 1: ST.pop()
                    section = Section(sline[3:], lvl=indent_level)
                    ST[-1].add_content(section)
                elif sline.startswith('### '):
                    while len(ST) > 1: ST.pop()
                    subsection = Subsection(sline[4:], lvl=indent_level)
                    ST[-1].add_content(subsection)
                elif sline.startswith('- '):
                    while ST[-1].lvl > indent_level or (not isinstance(ST[-1], ContentList) and not isinstance(ST[-1], UnorderedList)): ST.pop()
                    if isinstance(ST[-1], ContentList):    
                        ul = UnorderedList(lvl=indent_level)
                        ST[-1].add_content(ul)
                        ST.append(ul)
                        ul.add_item()
                        ST.append(ul.items[-1])
                    else:
                        assert isinstance(ST[-1], UnorderedList)
                        ul = ST[-1]
                        ul.add_item()
                        ST.append(ul.items[-1])
                elif bool(re.match(r'^\d+\. ', sline)):
                    while ST[-1].lvl > indent_level or (not isinstance(ST[-1], ContentList) and not isinstance(ST[-1], OrderedList)): ST.pop()
                    if isinstance(ST[-1], ContentList):    
                        ul = OrderedList(lvl=indent_level)
                        ST[-1].add_content(ul)
                        ST.append(ul)
                        ul.add_item()
                        ST.append(ul.items[-1])
                    else:
                        assert isinstance(ST[-1], OrderedList)
                        ul = ST[-1]
                        ul.add_item()
                        ST.append(ul.items[-1])
                elif sline == '\n':
                    if isinstance(ST[-1], Text):
                        ST.pop()
                else:
                    def clean() :
                        while ST[-1].lvl > indent_level or not isinstance(ST[-1], ContentList):
                            ST.pop()
                        assert isinstance(ST[-1], ContentList)
                        assert indent_level == ST[-1].lvl
                    if sline.startswith('$$'):
                        clean()
                        equation = Equation(lvl=indent_level)
                        ST[-1].add_content(equation)
                        ST.append(equation)
                    elif sline.startswith('```'):
                        clean()
                        codeblock = CodeBlock(lvl=indent_level)
                        ST[-1].add_content(codeblock)
                        ST.append(codeblock)
                    elif sline.startswith('![]'):
                        clean()
                        start = sline.find('(') + 1
                        end = sline.find(')')
                        image_path = sline[start:end]
                        scale_start = sline.find('width="') + 7
                        scale_end = sline.find('%"', scale_start)
                        scale = int(sline[scale_start:scale_end])
                        image = Image(image_path=image_path, scale=scale, lvl=indent_level)
                        ST[-1].add_content(image)
                    elif sline == "---\n":
                        clean()
                        separator = Separator(lvl=indent_level)
                        ST[-1].add_content(separator)
                    elif sline.startswith('!!! proof'):
                        clean()
                        proof = Proof(lvl=indent_level)
                        ST[-1].add_content(proof)
                        ST.append(proof)
                        ST.append(proof.contentlist)
                    elif sline.startswith('!!!'):
                        clean()
                        sline = re.sub(r'<a id=".*?"></a>', '', sline).strip()
                        parts = sline.split('"')[1].split(':')
                        name = parts[0].strip().split(' ')
                        chp_num = name[1].split('.')
                        chp = int(chp_num[0])
                        num = int(chp_num[1])
                        type = name[0]
                        assert(sline.split('"')[0].strip("!!!").strip() == type.lower())
                        title = parts[1].strip()
                        admonition = Admonition(chp=chp, num=num, type=type, title=title, lvl=indent_level)
                        ST[-1].add_content(admonition)
                        ST.append(admonition)
                        ST.append(admonition.contentlist)
                    else:
                        if isinstance(ST[-1], Text):
                            assert indent_level == ST[-1].lvl
                            ST[-1].append(sline)    
                        else:            
                            clean()
                            text = Text(text=sline, lvl=indent_level)
                            ST[-1].add_content(text)
                            ST.append(text)

            for i in range(len(ST)):
                print(f"type : {ST[i].__class__.__name__}, lvl : {ST[i].lvl}", end=" / ")
            print("\n")

        return contentlist
