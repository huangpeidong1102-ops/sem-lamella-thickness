"""Export reviewed measurements and their matching numbered image."""
import math
import os
import tempfile
from PIL import ImageDraw
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter


def export_selected(image, source_path, workbook_path, items, factor, unit, runmeta, scale_points, pending_count, region_count):
    image_path = os.path.splitext(workbook_path)[0] + '_标注.png'
    if not items or not math.isfinite(factor) or factor <= 0:
        raise ValueError('没有可导出的已确认测量或标尺无效。')
    image = image.copy()
    draw = ImageDraw.Draw(image)
    region = runmeta.get('roi')
    if region:
        if len(region) == 4 and all(isinstance(v, (int, float)) for v in region):
            draw.rectangle(region, outline='#00e9fa', width=2)
        else:
            draw.line([tuple(p) for p in region] + [tuple(region[0])], fill='#00e9fa', width=2)
    for poly in runmeta.get('exclusions', []):
        if len(poly) >= 3:
            draw.line([tuple(p) for p in poly] + [tuple(poly[0])], fill='#ff9a27', width=2)
    if len(scale_points) == 2:
        draw.line(scale_points, fill='cyan', width=2)
    book = Workbook()
    sheet = book.active
    sheet.title = '测量结果'
    sheet.append(['编号', '来源', '点1厚度', '点2厚度', '点3厚度', '三点均值', '单位', '点1起点X', '点1起点Y', '点1终点X', '点1终点Y', '点2起点X', '点2起点Y', '点2终点X', '点2终点Y', '点3起点X', '点3起点Y', '点3终点X', '点3终点Y', '有测点小于5像素', '测点位置重复', '计入统计', '阈值', '局部对比指标'])
    means=[]
    for i, item in enumerate(items, 1):
        points=item['points']; values=[float(p[2])*factor for p in points]
        centers=[((a[0]+b[0])/2, (a[1]+b[1])/2) for a,b,_ in points]
        duplicate=any(math.dist(centers[a], centers[b])<1 for a in range(3) for b in range(a+1,3))
        if not duplicate: means.append(sum(values)/3)
        for j,(a,b,px) in enumerate(points,1):
            draw.line([a,b],fill='red',width=2)
            draw.text(b,f'{i}.{j}:{float(px)*factor:.4g}',fill='yellow')
        draw.text(points[1][0],str(i),fill='yellow')
        sheet.append([i,item.get('source','自动'),*values,sum(values)/3,unit,*[float(v) for a,b,_ in points for p in (a,b) for v in p], '是' if any(p[2]<5 for p in points) else '否', '是' if duplicate else '否', '否' if duplicate else '是',item.get('threshold'),item.get('contrast_score')])
    sheet.freeze_panes='C2'
    sheet.auto_filter.ref=sheet.dimensions
    for cell in sheet[1]: cell.font=Font(bold=True)
    for col in range(1,25): sheet.column_dimensions[get_column_letter(col)].width=16
    meta=book.create_sheet('测量说明')
    for row in [('原图',os.path.basename(source_path)),('标尺单位',unit),('每像素长度',float(factor)),('选中导出数',len(items)),('有效片层数',len(means)),('片层均值的平均',sum(means)/len(means) if means else None),('片层均值的样本标准差',math.sqrt(sum((x-sum(means)/len(means))**2 for x in means)/(len(means)-1)) if len(means)>1 else None),('未审核候选数',pending_count),('待检查区域数',region_count),('相别',str(runmeta.get('phase',''))),('分析范围',str(runmeta.get('roi'))),('排除区',str(runmeta.get('exclusions',[]))),('说明','仅已确认且加入待导出的测量参与本次导出；重复测点不计统计。')]:
        meta.append(row)
        if isinstance(row[1],str): meta.cell(meta.max_row,2).data_type='s'
    meta.column_dimensions['A'].width=24;meta.column_dimensions['B'].width=70
    folder=os.path.dirname(os.path.abspath(workbook_path));temps=[]
    try:
        for suffix in ('.png','.xlsx'):
            fd,path=tempfile.mkstemp(prefix='.sem-export-',suffix=suffix,dir=folder);os.close(fd);temps.append(path)
        image.save(temps[0],format='PNG');book.save(temps[1])
        os.replace(temps[0],image_path);temps[0]=None
        os.replace(temps[1],workbook_path);temps[1]=None
    finally:
        for path in temps:
            if path and os.path.exists(path):os.unlink(path)
    return image_path
