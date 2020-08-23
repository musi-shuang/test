# _*_ utf-8 _*_

# 初入门的第一个学习文件

print("hello world!")

# 1.python中的标识符
"""
    在python中，所有标识符可以包括英文、数字以及下划线（_），但不能以数字开头。
python中的标识符是区分大小写的。
以下划线开头的标识符是有特殊意义的。
    以单下划线开头（_foo）的代表不能直接访问的类属性，需通过类提供的接口进行访问，不能用"fromxxx import *"而导入；
    以双下划线开头的（__foo）代表类的私有成员；
    以双下划线开头和结尾的（__foo__）代表python里特殊方法专用的标识，如__init__（）代表类的构造函数。

"""
# 2.python中的保留字

import keyword
print(keyword.kwlist)
"""
['False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 
'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']
"""

# 3.注释

"""
单行注释以#开头，多行注释可用多个#或者三个单引号或者三个双引号进行表示
"""

# 4.行与缩进
'''
使用代码块来代替大括号{}
'''
if True:
    print('True')
else:
    print('False')


# 5.使用反斜杠来拼接多行语句

# 6.输入（单个字符）若后续无程序，则退出
input('按下enter后退出')
print('lalala')

# 7.同一行多条语句，使用逗号分割
import sys; x = 'zhaoximou';sys.stdout.write(x + '\n' );

# 8.print默认换行，若要不换行，需要加上end = ''空字符串即可
x="a"
y="b"
# 换行输出
print( x )
print( y )
 
print('---------')
# 不换行输出
print( x, end=" " )
print( y, end=" " )
print()

# 9.import是导入模块from ... import ...是导入模块的函数，多个函数用逗号分隔
import sys
print('------------python导入模块---------------')
print('命令行参数是：')
for i in sys.argv:
    print(i)
print('\n python路径为：',sys.path)

from sys import argv,path
print('\n',path)
















