"""Persistent, pixel-verified projects for SEM measurement."""
import hashlib, json, math, os, tempfile
from PIL import Image
import numpy as np

def open_image(path):
    """Map high-bit-depth scalar images linearly into 8-bit for this detector."""
    with Image.open(path) as src:
        if src.mode.startswith('I;16') or src.mode in ('I','F'):
            values=np.asarray(src,dtype=np.float64)
            if not np.isfinite(values).all():raise ValueError('图像包含非有限灰度值。')
            lo=float(values.min());hi=float(values.max())
            gray=np.zeros(values.shape,dtype=np.uint8) if hi==lo else np.rint((values-lo)*(255.0/(hi-lo))).clip(0,255).astype(np.uint8)
            return Image.fromarray(gray).convert('RGB')
        return src.convert('RGB')

def native_pixel_digest(path):
    with Image.open(path) as src:
        h=hashlib.sha256();h.update(f'{src.width}x{src.height}:{src.mode}:'.encode('ascii'));h.update(src.tobytes());return h.hexdigest()


def pixel_digest(image):
    h=hashlib.sha256(); h.update(f'{image.width}x{image.height}:RGB:'.encode('ascii')); h.update(image.convert('RGB').tobytes()); return h.hexdigest()


def make_project(image, image_path, scale_points, scale, unit, roi, phase, params, runmeta, results, boundary="mask", exclusions=None, parallel_only=False, unpaired=None, neighbor_factor=6., auto_extra=False, uncertain_regions=None, rejected=None, auto_fine=True):
    def pack(items):
      records=[]
      for item in items:
        records.append({'points':[[[float(v) for v in a],[float(v) for v in b]] for a,b,_ in item['points']], 'source':item.get('source','自动'), 'coherence':float(item['coherence']) if item.get('coherence') is not None else None, 'contrast_score':float(item['contrast_score']) if item.get('contrast_score') is not None else None, 'contour':[[float(x),float(y)] for x,y in item.get('contour',[])], 'edge_methods':item.get('edge_methods'), 'neighbor_count':item.get('neighbor_count'), 'compact_neighbor_count':item.get('compact_neighbor_count'), 'threshold':item.get('threshold'), 'review_reason':item.get('review_reason')})
      return records
    records=pack(results);unpaired_records=pack(unpaired or [])
    roi_data=None if roi is None else {'kind':'polygon' if isinstance(roi,list) else 'rectangle','points':roi}
    mode='normalized' if pixel_digest(image)==pixel_digest(open_image(image_path)) else 'legacy'
    return {'version':1,'image_path':os.path.abspath(image_path),'pixel_sha256':pixel_digest(image),'native_pixel_sha256':native_pixel_digest(image_path),'pixel_mode':mode,'size':list(image.size),'scale_points':scale_points,'scale':scale,'unit':unit,'roi':roi_data,'exclusions':exclusions or [],'phase':phase,'boundary':boundary,'parallel_only':bool(parallel_only),'neighbor_factor':float(neighbor_factor),'auto_extra':bool(auto_extra),'auto_fine':bool(auto_fine),'params':params,'runmeta':runmeta,'results':records,'unpaired':unpaired_records,'rejected':pack(rejected or []),'uncertain_regions':uncertain_regions or []}


def save_project(path, data):
    if os.path.normcase(os.path.abspath(path))==os.path.normcase(os.path.abspath(data['image_path'])):raise ValueError('工程文件不能覆盖原始图片。')
    parent=os.path.dirname(os.path.abspath(path)); temp_path=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=parent,prefix='.sem-',suffix='.tmp',delete=False) as f:
            temp_path=f.name;json.dump(data,f,ensure_ascii=False,allow_nan=False,indent=2)
        os.replace(temp_path,path);temp_path=None
    finally:
        if temp_path and os.path.exists(temp_path):os.unlink(temp_path)


def load_project(path, image_path=None):
    with open(path,encoding='utf-8') as f:data=json.load(f)
    if data.get('version')!=1:raise ValueError('不支持的工程版本。')
    source=image_path or data['image_path'];image=open_image(source)
    if data.get('pixel_mode')=='legacy':
        with Image.open(source) as original:image=original.convert('RGB')
    if data.get('native_pixel_sha256') and data['native_pixel_sha256']!=native_pixel_digest(source):raise ValueError('原始图像像素与工程记录不一致。')
    if list(image.size)!=data['size'] or pixel_digest(image)!=data['pixel_sha256']:
        # Older projects used Pillow's direct RGB conversion for all images.
        if data.get('native_pixel_sha256'):raise ValueError('图片内容与工程记录不一致，不能继续加载。')
        with Image.open(source) as original:legacy=original.convert('RGB')
        if list(legacy.size)!=data['size'] or pixel_digest(legacy)!=data['pixel_sha256']:raise ValueError('图片内容与工程记录不一致，不能继续加载。')
        image=legacy
    w,h=image.size
    def point(p):
        if not isinstance(p,(list,tuple)) or len(p)!=2:raise ValueError('工程坐标格式错误。')
        x,y=map(float,p)
        if not math.isfinite(x) or not math.isfinite(y) or not (0<=x<w and 0<=y<h):raise ValueError('工程坐标越过图片范围。')
        return (x,y)
    cal=[point(p) for p in data['scale_points']]
    if len(cal) not in (0,2) or (cal and math.dist(*cal)<2):raise ValueError('工程标尺不正确。')
    if data['unit'] not in ('','µm','nm','mm'):raise ValueError('工程单位不正确。')
    roi=data.get('roi')
    if roi is not None:
        if roi.get('kind')=='polygon':
            roi=[point(p) for p in roi['points']]
            if len(roi)<3:raise ValueError('工程选区不正确。')
        elif roi.get('kind')=='rectangle':
            x0,y0,x1,y1=map(int,roi['points'])
            if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('工程选区不正确。')
            roi=(x0,y0,x1,y1)
        else:raise ValueError('工程选区类型不正确。')
    exclusions=[]
    for polygon in data.get('exclusions',[]):
        polygon=[point(p) for p in polygon]
        if len(polygon)<3:raise ValueError('工程排除区不正确。')
        exclusions.append(polygon)
    def unpack(source_rows):
      rows=[]
      for row in source_rows:
          if len(row['points'])!=3:raise ValueError('每个片层必须有三个测点。')
          pts=[]
          for a,b in row['points']:
              a,b=point(a),point(b); width=math.dist(a,b)
              if width<=0:raise ValueError('工程测点长度无效。')
              pts.append((a,b,width))
          score=row.get('coherence')
          if score is not None and (not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1):raise ValueError('方向得分无效。')
          contrast=row.get('contrast_score')
          if contrast is not None and (not isinstance(contrast,(int,float)) or not math.isfinite(contrast) or not -255<=contrast<=255):raise ValueError('局部对比记录无效。')
          contour=[point(p) for p in row.get('contour',[])]
          methods=row.get('edge_methods')
          if methods is not None and (len(methods)!=3 or any(len(pair)!=2 or any(v not in ('half','mask') for v in pair) for pair in methods)):raise ValueError('边界方法记录无效。')
          count=row.get('neighbor_count')
          if count is not None and (not isinstance(count,int) or isinstance(count,bool) or count<0):raise ValueError('邻近候选数无效。')
          compact=row.get('compact_neighbor_count')
          if compact is not None and (not isinstance(compact,int) or isinstance(compact,bool) or compact<0):raise ValueError('紧邻候选数无效。')
          threshold=row.get('threshold')
          if threshold is not None and (not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or not 0<=threshold<=255):raise ValueError('候选阈值无效。')
          reason=row.get('review_reason')
          if reason is not None and reason not in ('candidate','neighbor','sparse','threshold','fine'):raise ValueError('候选来源无效。')
          rows.append({'points':pts,'source':str(row.get('source','自动')),'coherence':score,'contrast_score':contrast,'contour':contour,'edge_methods':methods,'neighbor_count':count,'compact_neighbor_count':compact,'threshold':threshold,'review_reason':reason})
      return rows
    rows=unpack(data['results']);unpaired=unpack(data.get('unpaired',[]));rejected=unpack(data.get('rejected',[]))
    runmeta=data.get('runmeta')
    if runmeta is not None:
        if not isinstance(runmeta,dict) or runmeta.get('phase') not in ('暗相','亮相','手动') or not isinstance(runmeta.get('params'),dict):raise ValueError('工程测量来源记录不正确。')
        region=runmeta.get('roi')
        if region is not None:
            if not isinstance(region,(list,tuple)):raise ValueError('工程测量范围不正确。')
            if len(region)==4 and all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in region):
                x0,y0,x1,y1=map(float,region)
                if not all(math.isfinite(v) for v in (x0,y0,x1,y1)) or not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('工程测量范围超出图片。')
            else:
                if len(region)<3:raise ValueError('工程测量多边形不正确。')
                for p in region:point(p)
        polygons=runmeta.get('exclusions',[])
        if not isinstance(polygons,list):raise ValueError('工程测量排除区不正确。')
        for polygon in polygons:
            if not isinstance(polygon,(list,tuple)) or len(polygon)<3:raise ValueError('工程测量排除区不正确。')
            for p in polygon:point(p)
    regions=[]
    for entry in data.get('uncertain_regions',[]):
        box=entry.get('bbox')
        if not isinstance(box,(list,tuple)) or len(box)!=4 or any(not isinstance(v,int) or isinstance(v,bool) for v in box):raise ValueError('待辨区域坐标无效。')
        x0,y0,x1,y1=box
        if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('待辨区域超出图像。')
        score=entry.get('coherence');threshold=entry.get('threshold');state=entry.get('state','待检查');analysis_mode=entry.get('analysis_mode','standard')
        kind=entry.get('kind','component')
        if kind not in ('component','tile'):raise ValueError('审核区域类型无效。')
        if analysis_mode not in ('standard','fine') or (kind=='tile' and analysis_mode!='standard'):raise ValueError('待辨区域识别来源无效。')
        if kind=='component' and (not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1):raise ValueError('待辨区域方向评分无效。')
        if kind=='component' and (not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or not 0<=threshold<=255):raise ValueError('待辨区域阈值无效。')
        if kind=='tile' and (score is not None or threshold is not None):raise ValueError('分块不应具有检测评分。')
        if state not in ('待检查','已检查'):raise ValueError('待辨区域状态无效。')
        regions.append({'bbox':(x0,y0,x1,y1),'coherence':float(score) if score is not None else None,'threshold':float(threshold) if threshold is not None else None,'state':state,'kind':kind,'analysis_mode':analysis_mode})
    data.update({'scale_points':cal,'roi':roi,'exclusions':exclusions,'results':rows,'unpaired':unpaired,'rejected':rejected,'uncertain_regions':regions,'image_path':source})
    return data,image
