import re
import pdb
def norm_date(str_date):
    str_date = re.sub(r'([a-zA-Z].*?)\:|[a-zA-Z]', '', str_date)
    ls_num = re.findall(r'[0-9]{5,6}', str_date)
    for num in ls_num:
        yy = num[:4]
        hh = num[4:]
        sub_str = yy + ' ' + hh
        str_date = re.sub(num, sub_str, str_date)
    return str_date
# str_date = 'Date:02-04-201815:14:07'
# str_date = norm_date(str_date)
# print(str_date)
def norm_invoice(num_invoice):
    num_invoice = re.sub(r'\D', '', num_invoice)
    return num_invoice
# num_invoice = 'Invoicenumber:01000339450'
# num_invoice = norm_invoice(num_invoice)
# print(num_invoice)

