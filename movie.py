#coding=utf-8
#__author__ = 'lizk'

import omdb
import xlrd
import xlwt
data = xlrd.open_workbook('C:/Users/lizk/Desktop/2/movie/new.xlsx')
table = data.sheets()[0]
nrows = table.nrows
ncols = table.ncols
newfile = xlwt.Workbook()
wtable = newfile.add_sheet('title', cell_overwrite_ok=True)
d = {'id': 0, 'title': 1, 'year': 2, 'runtime': 3, 'director': 4, 'writer': 5,
     'actors': 6, 'language': 7, 'country': 8, 'awards': 9}
wtable.write(0, d['id'], 'id')
wtable.write(0, d['title'], 'title')
wtable.write(0, d['year'], 'year')
wtable.write(0, d['runtime'], 'runtime')
wtable.write(0, d['director'], 'director')
wtable.write(0, d['writer'], 'writer')
wtable.write(0, d['actors'], 'actors')
wtable.write(0, d['language'], 'language')
wtable.write(0, d['country'], 'country')
wtable.write(0, d['awards'], 'awards')
for rownum in range(table.nrows):
    imdbid = str(int(table.row(rownum)[1].value))
    need = 7 - len(imdbid)
    needlet ='0' * need
    id ='tt'+needlet+imdbid
    try:
        movie = omdb.imdbid(id)
    except:
        wtable.write(rownum+1, d['id'], imdbid)
        print "error id"+imdbid
        continue
    wtable.write(rownum+1, d['id'], imdbid)
    wtable.write(rownum+1, d['title'], movie.title)
    wtable.write(rownum+1, d['year'], movie.year)
    wtable.write(rownum+1, d['runtime'], movie.runtime)
    wtable.write(rownum+1, d['director'], movie.director)
    wtable.write(rownum+1, d['writer'], movie.writer)
    wtable.write(rownum+1, d['actors'], movie.actors)
    wtable.write(rownum+1, d['language'], movie.language)
    wtable.write(rownum+1, d['country'], movie.country)
    wtable.write(rownum+1, d['awards'], movie.awards)
    print movie.imdb_id
newfile.save('C:/Users/lizk/Desktop/2/movie/feature.xls')


