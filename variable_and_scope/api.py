import urllib
import re
import pdfkit

url="https://www.tensorflow.org/api_guides/python/train/"

page = urllib.urlopen(url).read()
keywords = re.findall("\"("+url+"[/\w]+)\"", page)
for keyword in keywords:
    print(keyword)
for keyword in keywords:
    pdfnamegp=re.search("(?<="+url+")[/\w]+",keyword)
    pdfname=pdfnamegp.group(0).replace("/", ".")+".pdf"
    #pdfkit.from_url(keyword, pdfname)