# coding:utf-8
import re    # python正则表达式模块数
#函数
def checkIdcard(idcard):
    #身份证的验证信息
    Errors=['验证通过!','身份证号码位数不对!',
            '身份证号码出生日期超出范围或含有非法字符!',
            '身份证号码校验错误!','身份证地区非法!']
    # 各个地区的身份证前两位
    area={"11":"北京","12":"天津","13":"河北","14":"山西",
          "15":"内蒙古","21":"辽宁","22":"吉林","23":"黑龙江",
          "31":"上海","32":"江苏","33":"浙江","34":"安徽","35":"福建",
          "36":"江西","37":"山东","41":"河南","42":"湖北","43":"湖南",
          "44":"广东","45":"广西","46":"海南","50":"重庆","51":"四川",
          "52":"贵州","53":"云南","54":"西藏","61":"陕西","62":"甘肃",
          "63":"青海","64":"宁夏","65":"新疆","71":"台湾","81":"香港",
          "82":"澳门","91":"国外"}
    idcard=str(idcard)  # 将idcard转为string类型
    idcard=idcard.strip()  # 默认删除空白符
    idcard_list=list(idcard)  # 将idcard 放入列表中
    idcard_brith = idcard[6:14]  # 生日字段
    idcard_sex = idcard[14:17]  # 性别字段
    year = idcard_brith[0:4]  # 年
    month = idcard_brith[4:6]  # 月
    day = idcard_brith[6:8]  # 日
    #地区校验，验证输入的前两位
    if(not area[(idcard)[0:2]]):
        print (Errors[4])
    #18位身份号码检测
    elif(len(idcard)==18):
        #出生日期的合法性检查
        if(int(idcard[6:10]) % 4 == 0 or (int(idcard[6:10]) % 100 == 0 and int(idcard[6:10])%4 == 0 )):
            # //闰年出生日期的合法性正则表达式
            ereg=re.compile('[1-9][0-9]{5}19[0-9]{2}((01|03|05|07|08|10|12)(0[1-9]|[1-2][0-9]|3[0-1])|(04|06|09|11)(0[1-9]|[1-2][0-9]|30)|02(0[1-9]|[1-2][0-9]))[0-9]{3}[0-9Xx]$')
        else:
            # //平年出生日期的合法性正则表达式
            ereg=re.compile('[1-9][0-9]{5}19[0-9]{2}((01|03|05|07|08|10|12)(0[1-9]|[1-2][0-9]|3[0-1])|(04|06|09|11)(0[1-9]|[1-2][0-9]|30)|02(0[1-9]|1[0-9]|2[0-8]))[0-9]{3}[0-9Xx]$')
        #//测试出生日期的合法性
        if(re.match(ereg,idcard)):
            #//计算校验位
            S = (int(idcard_list[0]) + int(idcard_list[10])) * 7 + \
                (int(idcard_list[1]) + int(idcard_list[11])) * 9 + \
                (int(idcard_list[2]) + int(idcard_list[12])) * 10 + \
                (int(idcard_list[3]) + int(idcard_list[13])) * 5 + \
                (int(idcard_list[4]) + int(idcard_list[14])) * 8 + \
                (int(idcard_list[5]) + int(idcard_list[15])) * 4 + \
                (int(idcard_list[6]) + int(idcard_list[16])) * 2 + \
                int(idcard_list[7]) * 1 + int(idcard_list[8]) * 6 + \
                int(idcard_list[9]) * 3
            Y = S % 11 # 求余
            JYM = "10X98765432"
            M = JYM[Y]#判断校验位
            if(M == idcard_list[17]):#检测ID的校验位
                print (Errors[0])
                print ('生日:'+year+'年'+ month +'月'+day+'日')
                if int(idcard_sex)%2 == 0:
                    print ('性别：女')
                else:
                    print ('性别：男')
            else:
                print (Errors[3])
        else:
            print (Errors[2])
    else:
        print (Errors[1])
if __name__ == '__main__':
    idcard ="110102197810272321"
    checkIdcard(idcard)