# 1.python中的变量
'''
    每个变量在使用前都需要赋值，变量赋值后才会被创建。变量没有类型，我们平时所说的类型是变量所指的内存中的对象的类型。=左侧是变量名，右侧是变量值
'''
counter = 100   # 整型变量
miles = 1000.6  # 浮点型变量
name = 'ximou'  # 字符串
print(counter,'\n',miles,'\n',name)

# 2.多变量赋值
a = b = c = 1
a,b,c = 1,2,'ximou'
print(a,'\n',b,'\n',c)

# 3.python3中六种基本变量类型：长度不可变：Number（数字）、String（字符串）、Tuple（元组）可变数据类型：List（列表）、Dictionary（字典）、Set（集合）
# Number:int(直接是长整形)、float、bool、complex（复数）
# 内置的 type() 函数可以用来查询变量所指的对象类型，也可以用isinstance来判断
# 可以使用del删除对象，del a,b,c
a, b, c, d = 20, 5.5, True, 4+3j
print(type(a), type(b), type(c), type(d))
print(isinstance(a,int))

# type与isinstance的区别是：type不会认为子类是父类的一种类型，而isinstance会
class  A:
    pass
class B(A):
    pass

print(isinstance(A(),A))
print(type(A()) == A)
print(isinstance(B(),A)) 
print(type(B()) == A)   # False

print('===============================')
# 4.数值运算
print(11 / 5) # 2.2真正的除法
print(11 // 5) # 2 取整

# 5.翻转字符串
def reverseWord(input):
    inputWords = input.split(' ')
    inputWords = inputWords[-1::-1]   # 步长为-1表示逆转
    # 重新组合字符串
    output = ' '.join(inputWords)
    return output

if __name__ == '__main__':
    input = 'shenghuo ya'
    new = reverseWord(input)
    print(new)





