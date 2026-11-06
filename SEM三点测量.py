import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np, csv, os, io, base64, html
from sem_analysis import detect, suggest_contrast_threshold, likely_same_lamella, find_uncertain_regions, review_tiles, enhance_sem_display, segment_inside_roi
from sem_project import make_project, save_project, load_project, open_image, native_pixel_digest, pixel_digest

def same_output_file(left,right):
 if not left or not right:return False
 if os.path.normcase(os.path.abspath(left))==os.path.normcase(os.path.abspath(right)):return True
 try:return os.path.samefile(left,right)
 except OSError:return False

def point_to_segment(x,y,a,b):
 dx,dy=b[0]-a[0],b[1]-a[1];den=dx*dx+dy*dy
 if den<=0:return float(np.hypot(x-a[0],y-a[1]))
 t=max(0.,min(1.,((x-a[0])*dx+(y-a[1])*dy)/den))
 return float(np.hypot(x-(a[0]+t*dx),y-(a[1]+t*dy)))

def inconsistent_manual_directions(points):
 """Flag strongly divergent intercepts; curved lamellae can also trigger this."""
 vectors=[np.asarray(b,dtype=float)-np.asarray(a,dtype=float) for a,b,_ in points]
 units=[v/np.linalg.norm(v) for v in vectors]
 return any(abs(float(units[i]@units[j]))<np.cos(np.deg2rad(45)) for i in range(3) for j in range(i+1,3))

def boundary_fallback(item,runmeta):
 methods=item.get('edge_methods')
 return bool(runmeta and runmeta.get('boundary')=='half' and methods and any(end=='mask' for pair in methods for end in pair))

def weak_local_contrast(item):
 value=item.get('contrast_score')
 return value is not None and value<10

def measurement_outside(item,mask):
 return any(not segment_inside_roi(a,b,mask) for a,b,_ in item['points'])

def repeated_measurement_positions(points):
 centers=[((a[0]+b[0])/2,(a[1]+b[1])/2) for a,b,_ in points]
 return any(np.hypot(centers[i][0]-centers[j][0],centers[i][1]-centers[j][1])<1 for i in range(3) for j in range(i+1,3))

def duplicate_references(item,pending,prior_count,results,limit=3):
 found=[]
 for i,other in enumerate(results,1):
  if likely_same_lamella(item,other):found.append(f'已确认{i}')
  if len(found)>limit:return found[:limit]+['等']
 for i,other in enumerate(pending[:prior_count],1):
  if other.get('threshold')!=item.get('threshold') and likely_same_lamella(item,other):found.append(f'待审{i}')
  if len(found)>limit:return found[:limit]+['等']
 return found

class App:
 def __init__(self,root):
  self.root=root;root.title('SEM 片层厚度测量 · 三点法');root.geometry('1400x850');self.im=None;self.results=[];self.roi=None;self.scalepts=[];self.mode='';self.pending=[];self.zoom=1;self.photo=None;self.path='';self.runmeta=None;self.zoom_factor=1;self.edit_target=None;self.exclusions=[];self.unpaired=[];self.rejected=[];self.preview_unpaired=None;self.review_window=None;self.uncertain_regions=[];self.preview_region=None;self.region_window=None;self.boundary=tk.StringVar(value='half');self.parallel_only=tk.BooleanVar(value=True);self.neighbor_factor=6.;self.show_review=tk.BooleanVar(value=True);self.show_regions=tk.BooleanVar(value=True);self.auto_extra=tk.BooleanVar(value=True)
  menu=tk.Menu(root);files=tk.Menu(menu,tearoff=0);files.add_command(label='打开图片',command=self.open);files.add_command(label='保存工程',command=self.save_session);files.add_command(label='打开工程',command=self.load_session);menu.add_cascade(label='文件',menu=files);boundary_menu=tk.Menu(menu,tearoff=0);boundary_menu.add_radiobutton(label='局部半高边界',variable=self.boundary,value='half',command=self.boundary_changed);boundary_menu.add_radiobutton(label='分割轮廓边界（旧版）',variable=self.boundary,value='mask',command=self.boundary_changed);menu.add_cascade(label='边界方法',menu=boundary_menu);settings=tk.Menu(menu,tearoff=0);settings.add_command(label='估算对比阈值',command=self.suggest_threshold);settings.add_separator();settings.add_checkbutton(label='优先排列邻近平行候选',variable=self.parallel_only,command=self.parallel_changed);settings.add_command(label='调整平行邻距范围',command=self.set_neighbor_factor);settings.add_command(label='复核待选候选',command=self.open_review);settings.add_command(label='检查无法测量的区域',command=self.open_regions);settings.add_command(label='导出待审核标注图',command=self.export_review);settings.add_command(label='用其他阈值寻找待复核候选',command=self.explore_threshold);settings.add_checkbutton(label='识别时自动补充其他阈值候选',variable=self.auto_extra,command=self.auto_extra_changed);settings.add_separator();settings.add_command(label='绘制排除区',command=lambda:self.setmode('exclude'));settings.add_command(label='完成排除区',command=self.finish_exclusion);settings.add_command(label='清空所有排除区',command=self.clear_exclusions);menu.add_cascade(label='工具',menu=settings);root.config(menu=menu)
  self.rejected_window=None;self.source_digest=None;self.display_mode=tk.StringVar(value='原图');self.display_cache={};self.auto_fine=tk.BooleanVar(value=True);settings.add_command(label='找回已丢弃的候选',command=self.open_rejected);settings.add_command(label='设置细小组织候选参数',command=self.fine_preset);settings.add_checkbutton(label='默认参数无候选时补充细小候选',variable=self.auto_fine);settings.add_command(label='导出伪彩增强图片',command=self.export_color_preview)
  side=ttk.Frame(root,padding=10,width=300);side.pack(side='left',fill='y');side.pack_propagate(False)
  area=ttk.Frame(root);area.pack(side='right',fill='both',expand=True)
  ttk.Label(side,text='SEM 片层厚度测量',font=('Microsoft YaHei',16,'bold')).pack(anchor='w');ttk.Label(side,text='导入 → 校准 → 选区 → 识别 → 核对',wraplength=280).pack(anchor='w',pady=6)
  for label,fn in [('打开 SEM 图片（支持 TIFF）',self.open),('① 标尺两端校准',lambda:self.setmode('scale')),('② 框选片层区域',lambda:self.setmode('roi')),('多边形选区：依次点击顶点',lambda:self.setmode('polygon')),('完成多边形选区',self.finish_polygon),('取消选区（使用全图）',self.resetroi)]: ttk.Button(side,text=label,command=fn).pack(fill='x',pady=2)
  row=ttk.Frame(side);row.pack(fill='x');ttk.Label(row,text='标尺长度').pack(side='left');self.scale=tk.StringVar(value='');ttk.Entry(row,textvariable=self.scale,width=8).pack(side='left');self.unit=tk.StringVar(value='');ttk.Combobox(row,textvariable=self.unit,values=['µm','nm','mm'],width=5,state='readonly').pack(side='left')
  self.phase=tk.StringVar(value='暗相');self.entries={}
  for name,default in [('目标相','暗相'),('最小厚度（像素）','2.5'),('最大厚度（像素）','20'),('最短长度（像素）','30'),('最小长宽比','4'),('对比阈值（0–255）','1'),('方向一致性（0–1）','0.7')]:
   r=ttk.Frame(side);r.pack(fill='x',pady=3);ttk.Label(r,text=name).pack(side='left')
   if name=='目标相': ttk.Combobox(r,textvariable=self.phase,values=['暗相','亮相'],width=8,state='readonly').pack(side='right')
   else:
    v=tk.StringVar(value=default);self.entries[name]=v;ttk.Entry(r,textvariable=v,width=9).pack(side='right')
  ttk.Button(side,text='③ 识别候选（三点待审核）',command=self.run).pack(fill='x',pady=7)
  ttk.Button(side,text='手动补充一个片层的三点',command=lambda:self.setmode('manual')).pack(fill='x',pady=2)
  ttk.Button(side,text='④ 点击图片剔除误识别',command=lambda:self.setmode('remove')).pack(fill='x',pady=2)
  ttk.Button(side,text='导出测量 CSV 与标注图',command=self.export).pack(fill='x',pady=4)
  self.summary=tk.StringVar(value='尚无统计结果');ttk.Label(side,textvariable=self.summary,wraplength=280).pack(fill='x',pady=4)
  self.info=tk.StringVar(value='请选择图片。');ttk.Label(side,textvariable=self.info,wraplength=280,foreground='#245c62').pack(fill='x',pady=8)
  ttk.Label(side,text='橙色为待审核候选，不计入统计；红色为已确认。全部相界及测点均须核对。',wraplength=280,foreground='#596b70').pack(fill='x',pady=6)
  bar=ttk.Frame(area);bar.pack(fill='x');self.review_count=tk.StringVar(value='待复核 0')
  for label,factor in [('放大',1.5),('缩小',1/1.5),('适合窗口',0)]:ttk.Button(bar,text=label,command=lambda f=factor:self.setzoom(f)).pack(side='left')
  ttk.Button(bar,textvariable=self.review_count,command=self.open_review).pack(side='left',padx=4)
  ttk.Checkbutton(bar,text='显示待审核',variable=self.show_review,command=self.render).pack(side='left')
  ttk.Button(bar,text='修正选中片层三点',command=self.edit_selected).pack(side='left',padx=4)
  ttk.Button(bar,text='删除选中片层',command=self.remove_selected).pack(side='left',padx=4)
  ttk.Label(bar,text='滚动条移动图片；Esc 取消操作').pack(side='left',padx=5)
  regionbar=ttk.Frame(area);regionbar.pack(fill='x');self.region_count=tk.StringVar(value='待辨区域 0')
  ttk.Button(regionbar,textvariable=self.region_count,command=self.open_regions).pack(side='left',padx=4)
  ttk.Checkbutton(regionbar,text='显示待辨区域（蓝框）',variable=self.show_regions,command=self.render).pack(side='left')
  ttk.Label(regionbar,text='空闲时点击橙线或蓝框可定位审核').pack(side='left',padx=5)
  ttk.Label(regionbar,text='显示').pack(side='left',padx=(8,2));preview=ttk.Combobox(regionbar,textvariable=self.display_mode,values=['原图','灰度增强','伪彩增强'],state='readonly',width=9);preview.pack(side='left');preview.bind('<<ComboboxSelected>>',lambda e:self.change_display())
  viewport=ttk.Frame(area);viewport.pack(fill='both',expand=True);viewport.rowconfigure(0,weight=1);viewport.columnconfigure(0,weight=1)
  self.canvas=tk.Canvas(viewport,bg='#243642',highlightthickness=0);self.canvas.grid(row=0,column=0,sticky='nsew')
  sx=ttk.Scrollbar(viewport,orient='horizontal',command=self.canvas.xview);sx.grid(row=1,column=0,sticky='ew')
  sy=ttk.Scrollbar(viewport,orient='vertical',command=self.canvas.yview);sy.grid(row=0,column=1,sticky='ns')
  self.canvas.configure(xscrollcommand=sx.set,yscrollcommand=sy.set);self.canvas.bind('<Button-1>',self.click);self.canvas.bind('<Configure>',lambda e:self.render());root.bind('<Escape>',lambda e:self.cancel());root.protocol('WM_DELETE_WINDOW',self.close)
  self.table=ttk.Treeview(area,columns=('id','a','b','c','mean','flag'),show='headings',height=7)
  for col,label in zip(('id','a','b','c','mean','flag'),('片层编号','第1点','第2点','第3点','片层平均','核对提示')): self.table.heading(col,text=label);self.table.column(col,width=110)
  self.table.pack(fill='x');self.table.bind('<<TreeviewSelect>>',lambda e:self.render());self.scale.trace_add('write',lambda *_:self.refresh());self.unit.trace_add('write',lambda *_:self.refresh())
 def close(self):
  if (self.results or self.unpaired or self.uncertain_regions) and not messagebox.askyesno('退出软件','未保存的测量将丢失。请先保存工程；确定退出吗？'):return
  self.root.destroy()
 def open(self):
  if (self.results or self.unpaired or self.uncertain_regions) and not messagebox.askyesno('替换测量','打开新图片会清空当前测量。请先保存工程或导出结果；继续吗？'):return
  path=filedialog.askopenfilename(filetypes=[('显微图片','*.tif *.tiff *.png *.jpg *.jpeg *.bmp'),('全部文件','*.*')])
  if not path:return
  try:
   im=open_image(path);digest=native_pixel_digest(path)
  except Exception as e:messagebox.showerror('无法打开',str(e));return
  self.im=im;self.display_cache={};self.display_mode.set('原图');self.source_digest=digest;self.source_rgb_digest=pixel_digest(im)
  self.close_review();self.close_regions();self.unpaired=[];self.rejected=[];self.uncertain_regions=[];self.zoom_factor=1;self.path=path;self.results=[];self.runmeta=None;self.roi=None;self.exclusions=[];self.scalepts=[];self.mode='';self.pending=[];self.scale.set('');self.unit.set('');self.refresh();self.info.set('图片已打开。请标记标尺两端并输入该图的实际长度。')
 def edit_selected(self):
  selected=self.table.selection()
  if not selected:self.info.set('请先在下方列表中选中要修正的片层。');return
  idx=self.table.index(selected[0]);self.edit_target=self.results[idx];self.mode='edit';self.pending=[]
  self.info.set('修正片层：依次点击三个厚度截距的两端，共六次；按 Esc 可取消。');self.render()
 def preview_image(self,mode):
  if mode=='原图':return self.im
  if mode not in self.display_cache:self.display_cache[mode]=enhance_sem_display(self.im,color=(mode=='伪彩增强'))
  return self.display_cache[mode]
 def change_display(self):
  if self.im is None:self.display_mode.set('原图');self.info.set('请先打开图片。');return
  try:self.preview_image(self.display_mode.get());self.render();self.info.set('显示已切换；识别和厚度计算始终使用打开时的原始图像。伪彩颜色不代表不同物相。')
  except Exception as e:self.display_mode.set('原图');self.render();messagebox.showerror('显示增强失败',str(e))
 def export_color_preview(self):
  if self.im is None:self.info.set('请先打开图片。');return
  path=filedialog.asksaveasfilename(defaultextension='.png',filetypes=[('PNG 图片','*.png')])
  if not path:return
  if same_output_file(path,self.path):messagebox.showerror('无法保存','增强图片不能覆盖原始图片。');return
  try:self.preview_image('伪彩增强').save(path,format='PNG')
  except Exception as e:messagebox.showerror('导出失败',str(e));return
  self.info.set('伪彩增强图已导出；颜色不代表物相，测量使用原图。')
 def roi_from(self,region,exclusions):
  mask=Image.new('1',(self.im.width,self.im.height),0);d=ImageDraw.Draw(mask)
  if region:
   if len(region)==4 and all(isinstance(v,(int,float)) for v in region):
    x0,y0,x1,y1=region;d.rectangle((x0,y0,x1-1,y1-1),fill=1)
   else:d.polygon(region,fill=1)
  else:d.rectangle((0,0,self.im.width-1,int(self.im.height*.91)-1),fill=1)
  for poly in exclusions:d.polygon(poly,fill=0)
  return np.asarray(mask,dtype=bool)
 def analysis_roi(self):return self.roi_from(self.roi,self.exclusions)
 def suggest_threshold(self):
  if self.im is None:self.info.set('请先打开图片。');return
  try:value,sigma=suggest_contrast_threshold(self.im,self.analysis_roi())
  except ValueError as e:messagebox.showwarning('无法估算',str(e));return
  self.entries['对比阈值（0–255）'].set(str(value));self.info.set(f'高频波动估计 {sigma:.1f} 灰度级，建议阈值 {value:g}；重新识别并核对。')
 def auto_extra_changed(self):
  self.info.set('自动补充不同阈值候选已开启；重新识别后生效。' if self.auto_extra.get() else '自动补充不同阈值候选已关闭；重新识别后生效。')
 def set_neighbor_factor(self):
  v=simpledialog.askfloat('平行邻距范围','输入邻距的倍宽系数（2–20，默认 6）。增大后间距较大的候选更可能排在前面；所有候选都需审核。',initialvalue=self.neighbor_factor,minvalue=2,maxvalue=20,parent=self.root)
  if v is None:return
  if not np.isfinite(v) or not 2<=v<=20:messagebox.showwarning('参数无效','请输入 2 到 20 之间的有限数值。');return
  self.neighbor_factor=float(v);self.info.set(f'邻距倍宽系数已设为 {v:g}；重新识别后生效。')
 def parallel_changed(self):
  self.info.set('将优先排列邻近平行候选；所有候选仍需审核。重新识别后生效。' if self.parallel_only.get() else '将按识别顺序排列；所有候选仍需审核。重新识别后生效。')
 def fine_preset(self):
  if not messagebox.askyesno('细小组织候选参数','将设置厚度 1–8 像素、最短长度 20 像素、长宽比 3、方向一致性 0.5。少于 5 像素的厚度可能不可靠，建议使用更高分辨率图片。参数仅在重新识别后生效；重新识别会替换当前结果，请先保存工程。继续吗？'):return
  for key,value in [('最小厚度（像素）','1'),('最大厚度（像素）','8'),('最短长度（像素）','20'),('最小长宽比','3'),('方向一致性（0–1）','0.5')]:self.entries[key].set(value)
  self.info.set('细小组织参数已设置；重新识别后生效。小于 5 像素的测点需重点审核。')
 def explore_threshold(self):
  if self.im is None or not self.runmeta or self.runmeta.get('phase') not in ('暗相','亮相'):
   self.info.set('请先完成一次自动识别，再探索其他阈值。');return
  threshold=simpledialog.askfloat('探索其他阈值','输入 0–255 的其他对比阈值；新候选仅进入待复核，不自动计入统计。',minvalue=0,maxvalue=255,parent=self.root)
  if threshold is None:return
  if not np.isfinite(threshold) or not 0<=threshold<=255:messagebox.showwarning('阈值无效','请输入 0 到 255 之间的有限数值。');return
  try:
   params=self.runmeta['params'];mi=float(params['最小厚度（像素）']);ma=float(params['最大厚度（像素）']);length=float(params['最短长度（像素）']);ratio=float(params['最小长宽比']);coh=float(params['方向一致性（0–1）'])
   roi=self.roi_from(self.runmeta.get('roi'),self.runmeta.get('exclusions',[]))
   self.info.set('正在用其他阈值寻找待复核候选……');self.root.update_idletasks()
   rows=detect(self.im,roi,'dark' if self.runmeta['phase']=='暗相' else 'bright',mi,ma,length,ratio,threshold,coherence_min=coh,boundary_method=self.runmeta.get('boundary','mask'),neighbor_width_factor=float(self.runmeta.get('neighbor_factor',6.)))
   found_regions=find_uncertain_regions(self.im,roi,'dark' if self.runmeta['phase']=='暗相' else 'bright',mi,ma,length,threshold,rows,coherence_min=coh)
  except Exception as e:messagebox.showerror('探索失败',str(e));return
  def key(item):
   return tuple(round(float(v),2) for a,b,_ in item['points'] for v in (*a,*b))
  baseline=self.runmeta['params'].get('对比阈值（0–255）')
  def item_threshold(item):
   t=item.get('threshold')
   if t is None and item.get('source','自动') in ('自动','人工接纳'):
    try:t=float(baseline)
    except (TypeError,ValueError):pass
   return t
  known={(item_threshold(x),key(x)) for x in self.results+self.unpaired+self.rejected}
  new=[]
  for item in rows:
   k=(float(threshold),key(item))
   if k in known:continue
   known.add(k);item['review_reason']='threshold';new.append(item)
  new.sort(key=lambda item:any(likely_same_lamella(item,existing) for existing in self.results+self.unpaired))
  region_keys={(x.get('analysis_mode','standard'),x.get('threshold'),tuple(x['bbox'])) for x in self.uncertain_regions if x.get('kind')!='tile'}
  new_regions=[]
  for region in found_regions:
   k=('standard',float(threshold),tuple(region['bbox']))
   if k in region_keys:continue
   region_keys.add(k);region['state']='待检查';new_regions.append(region)
  self.uncertain_regions.extend(new_regions)
  self.unpaired.extend(new);self.review_count.set(f'待复核 {len(self.unpaired)}')
  if self.review_window is not None:self.update_review()
  self.update_regions();self.refresh()
  self.info.set(f'阈值 {threshold:g} 新增 {len(new)} 条待复核、{len(new_regions)} 个待辨区域；正式结果未改变。请检查疑似重复项。')
 def boundary_changed(self):
  if hasattr(self,'info'):self.info.set('已选择局部半高边界。需重新识别后生效。' if self.boundary.get()=='half' else '已选择分割轮廓边界。需重新识别后生效。')
 def save_session(self):
  if self.im is None or not self.path:self.info.set('请先打开图片。');return
  path=filedialog.asksaveasfilename(defaultextension='.sem.json',filetypes=[('SEM 测量工程','*.sem.json'),('JSON','*.json')])
  if not path:return
  try:
   if self.source_digest is not None and (native_pixel_digest(self.path)!=self.source_digest or pixel_digest(open_image(self.path))!=self.source_rgb_digest):raise ValueError('原图像素已在软件外发生变化。请在原路径恢复原图，避免工程关联错误。')
   obj=make_project(self.im,self.path,self.scalepts,self.scale.get(),self.unit.get(),self.roi,self.phase.get(),{k:v.get() for k,v in self.entries.items()},self.runmeta,self.results,boundary=self.boundary.get(),exclusions=self.exclusions,parallel_only=self.parallel_only.get(),unpaired=self.unpaired,neighbor_factor=self.neighbor_factor,auto_extra=self.auto_extra.get(),uncertain_regions=self.uncertain_regions,rejected=self.rejected,auto_fine=self.auto_fine.get());save_project(path,obj)
  except Exception as e:messagebox.showerror('保存工程失败',str(e));return
  self.info.set('工程已保存，之后可继续核对和修正。')
 def load_session(self):
  path=filedialog.askopenfilename(filetypes=[('SEM 测量工程','*.sem.json *.json')])
  if not path:return
  try:
   try:data,im=load_project(path)
   except FileNotFoundError:
    source=filedialog.askopenfilename(title='请选择工程对应的原始图片',filetypes=[('显微图片','*.tif *.tiff *.png *.jpg *.jpeg *.bmp')])
    if not source:return
    data,im=load_project(path,source)
   if not isinstance(data['params'],dict) or data['phase'] not in ('暗相','亮相') or data.get('boundary','mask') not in ('half','mask'):raise ValueError('工程参数不正确。')
   nf=float(data.get('neighbor_factor',6.))
   if not np.isfinite(nf) or not 2<=nf<=20:raise ValueError('工程邻距倍数不正确。')
   if (data['results'] or data['unpaired'] or data['uncertain_regions']) and not isinstance(data.get('runmeta'),dict):raise ValueError('工程缺少测量来源记录。')
  except Exception as e:messagebox.showerror('打开工程失败',str(e));return
  if (self.results or self.unpaired or self.uncertain_regions) and not messagebox.askyesno('替换测量','打开工程会替换当前测量。请先保存当前工程；继续吗？'):return
  self.close_review();self.close_regions();self.display_cache={};self.display_mode.set('原图');self.uncertain_regions=data['uncertain_regions'];self.unpaired=data['unpaired'];self.rejected=data['rejected'];self.im=im;self.path=data['image_path'];self.source_digest=native_pixel_digest(self.path);self.source_rgb_digest=pixel_digest(open_image(self.path));self.results=[];self.scalepts=data['scale_points'];self.roi=data['roi'];self.exclusions=data['exclusions'];self.scale.set(str(data['scale']));self.unit.set(data['unit']);self.phase.set(data['phase']);self.boundary.set(data.get('boundary','mask'));self.parallel_only.set(bool(data.get('parallel_only',False)));self.neighbor_factor=nf;self.auto_extra.set(bool(data.get('auto_extra',False)))
  self.auto_fine.set(bool(data.get('auto_fine',True)))
  for k,v in self.entries.items():
   if k in data['params']:v.set(str(data['params'][k]))
  legacy=[x for x in data['results'] if x.get('source','自动')=='自动']
  for item in legacy:
   if item.get('review_reason') is None:item['review_reason']='candidate'
  self.results=[x for x in data['results'] if x.get('source','自动')!='自动'];self.unpaired=legacy+self.unpaired
  self.runmeta=data.get('runmeta');self.display_cache={};self.display_mode.set('原图');self.pending=[];self.mode='';self.edit_target=None;self.zoom_factor=1;self.refresh()
  self.info.set(f'工程已恢复；{len(legacy)} 条旧版自动结果转为待审核。' if legacy else '工程已恢复；图片内容已核对。')
 def pending_regions(self):return [x for x in self.uncertain_regions if x.get('state','待检查')=='待检查']
 def close_regions(self):
  if self.region_window is not None:
   self.region_window.destroy();self.region_window=None
  self.preview_region=None
 def open_regions(self):
  if not self.uncertain_regions:self.info.set('没有待辨区域；紫框内未标区域仍需人工检查。');return
  if self.region_window is not None:self.region_window.lift();return
  win=tk.Toplevel(self.root);self.region_window=win;win.title('待辨区域（没有自动厚度）');win.geometry('440x440')
  ttk.Label(win,text='蓝框为可疑区域；分块覆盖整个分析范围，供逐块检查漏检。分块不代表检测到了片层。',wraplength=410).pack(fill='x',padx=10,pady=8)
  self.region_filter=tk.StringVar(value='全部');self.region_selector=ttk.Combobox(win,textvariable=self.region_filter,state='readonly');self.region_selector.pack(fill='x',padx=10);self.region_selector.bind('<<ComboboxSelected>>',lambda e:self.update_regions())
  frame=ttk.Frame(win);frame.pack(fill='both',expand=True,padx=10);scroll=ttk.Scrollbar(frame,orient='vertical');self.region_list=tk.Listbox(frame,exportselection=False,yscrollcommand=scroll.set);scroll.config(command=self.region_list.yview);scroll.pack(side='right',fill='y');self.region_list.pack(side='left',fill='both',expand=True)
  self.region_list.bind('<<ListboxSelect>>',lambda e:self.region_preview())
  buttons=ttk.Frame(win);buttons.pack(fill='x',padx=10,pady=8)
  ttk.Button(buttons,text='手动测量',command=self.measure_region).pack(side='left')
  self.region_check_button=ttk.Button(buttons,text='标记已检查',command=self.check_region);self.region_check_button.pack(side='left',padx=5)
  ttk.Button(buttons,text='关闭',command=self.dismiss_regions).pack(side='right')
  win.protocol('WM_DELETE_WINDOW',self.dismiss_regions);self.update_regions()
 def update_regions(self):
  pending=self.pending_regions();self.region_count.set(f'待辨区域 {len(pending)}')
  if self.region_window is None:return
  options=['全部','全范围分块','细小补充区域','已检查']+[f'阈值 {t:.12g}' for t in sorted({item['threshold'] for item in pending if item.get('kind')!='tile'})];self.region_selector['values']=options
  if self.region_filter.get() not in options:self.region_filter.set('全部')
  choice=self.region_filter.get();self.region_check_button.config(text='撤销已检查' if choice=='已检查' else '标记已检查')
  if choice=='已检查':self.region_display=[(i,item) for i,item in enumerate(self.uncertain_regions,1) if item.get('state')=='已检查']
  else:self.region_display=[(i,item) for i,item in enumerate(pending,1) if choice=='全部' or (choice=='全范围分块' and item.get('kind')=='tile') or (choice=='细小补充区域' and item.get('analysis_mode')=='fine') or (item.get('kind')!='tile' and choice==f"阈值 {item['threshold']:.12g}")]
  self.region_list.delete(0,'end')
  for i,item in self.region_display:
   x0,y0,x1,y1=item['bbox'];kind='全范围分块' if item.get('kind')=='tile' else f"{'细小补充区域' if item.get('analysis_mode')=='fine' else '可疑区域'}（阈值 {item['threshold']:g}）";self.region_list.insert('end',f'{kind} {i}  原图坐标 ({x0},{y0})–({x1},{y1})')
  self.preview_region=None;self.render()
 def region_preview(self):
  if self.region_window is None:return
  selection=self.region_list.curselection();display=self.region_display
  if not selection or selection[0]>=len(display):self.preview_region=None;self.render();return
  self.preview_region=display[selection[0]][1];self.render();x0,y0,x1,y1=self.preview_region['bbox'];x=(x0+x1)/2;y=(y0+y1)/2
  w=max(self.canvas.winfo_width(),self.im.width*self.zoom);h=max(self.canvas.winfo_height(),self.im.height*self.zoom)
  self.canvas.xview_moveto(max(0,(self.ox+x*self.zoom-self.canvas.winfo_width()/2)/w));self.canvas.yview_moveto(max(0,(self.oy+y*self.zoom-self.canvas.winfo_height()/2)/h))
 def dismiss_regions(self):self.close_regions();self.render()
 def check_region(self):
  if self.region_window is None or not self.region_list.curselection():self.info.set('请先选中要检查的区域。');return
  item=self.region_display[self.region_list.curselection()[0]][1];undo=item.get('state')=='已检查';item['state']='待检查' if undo else '已检查';self.update_regions();self.refresh();self.info.set('已撤销检查状态，区域重新进入待检查。' if undo else '区域已标记为检查过；不会自动生成或删除厚度。')
 def measure_region(self):
  if self.region_window is None or not self.region_list.curselection():self.info.set('请先选中要测量的区域。');return
  item=self.region_display[self.region_list.curselection()[0]][1];self.close_regions();self.setmode('manual');self.preview_region=item;self.render();self.info.set('请在黄色框附近手动点击三个截距的六个端点；区域不会自动标为已检查。')
 def close_review(self):
  self.close_rejected()
  if self.review_window is not None:
   self.review_window.destroy();self.review_window=None
  self.preview_unpaired=None
 def close_rejected(self):
  if self.rejected_window is not None:
   self.rejected_window.destroy();self.rejected_window=None
  self.preview_unpaired=None
 def open_rejected(self):
  if not self.rejected:self.info.set('没有已丢弃的候选。');return
  if self.rejected_window is not None:self.rejected_window.lift();return
  self.close_review();self.close_regions()
  win=tk.Toplevel(self.root);self.rejected_window=win;win.title('已丢弃的候选');win.geometry('440x410')
  ttk.Label(win,text='选中后黄色标出三个测点。找回后仍须审核，不会自动计入统计。',wraplength=410).pack(fill='x',padx=10,pady=8)
  frame=ttk.Frame(win);frame.pack(fill='both',expand=True,padx=10);scroll=ttk.Scrollbar(frame,orient='vertical');self.rejected_list=tk.Listbox(frame,exportselection=False,yscrollcommand=scroll.set);scroll.config(command=self.rejected_list.yview);scroll.pack(side='right',fill='y');self.rejected_list.pack(side='left',fill='both',expand=True)
  for i,item in enumerate(self.rejected,1):
   t=item.get('threshold');label=f'{t:g}' if t is not None else '未记录';a,b,_=item['points'][1];self.rejected_list.insert('end',f'候选 {i}  阈值 {label}  原图坐标 ({(a[0]+b[0])/2:.0f}, {(a[1]+b[1])/2:.0f})')
  self.rejected_list.bind('<<ListboxSelect>>',lambda e:self.rejected_preview())
  ttk.Button(win,text='找回选中候选',command=self.restore_rejected).pack(pady=8)
  win.protocol('WM_DELETE_WINDOW',self.dismiss_rejected)
 def rejected_preview(self):
  if self.rejected_window is None or not self.rejected_list.curselection():return
  self.preview_unpaired=self.rejected[self.rejected_list.curselection()[0]];self.render()
  pts=self.preview_unpaired['points'];x=sum((a[0]+b[0])/2 for a,b,_ in pts)/3;y=sum((a[1]+b[1])/2 for a,b,_ in pts)/3
  w=max(self.canvas.winfo_width(),self.im.width*self.zoom);h=max(self.canvas.winfo_height(),self.im.height*self.zoom)
  self.canvas.xview_moveto(max(0,(self.ox+x*self.zoom-self.canvas.winfo_width()/2)/w));self.canvas.yview_moveto(max(0,(self.oy+y*self.zoom-self.canvas.winfo_height()/2)/h))
 def dismiss_rejected(self):self.close_rejected();self.render()
 def restore_rejected(self):
  if self.rejected_window is None or not self.rejected_list.curselection():self.info.set('请先选择已丢弃候选。');return
  item=self.rejected.pop(self.rejected_list.curselection()[0]);self.unpaired.append(item);self.close_rejected();self.refresh();self.info.set('候选已找回待复核；尚未计入统计。');self.open_review();self.review_list.selection_set(len(self.unpaired)-1);self.review_list.see(len(self.unpaired)-1);self.review_preview()
 def open_review(self):
  if not self.unpaired:self.info.set('没有待复核的候选。');return
  self.close_rejected()
  if self.review_window is not None:
   self.review_window.lift();return
  win=tk.Toplevel(self.root);self.review_window=win;win.title('待复核候选');win.geometry('400x410')
  ttk.Label(win,text='候选未计入统计。划痕、成像条纹或噪声也可能被识别；请结合原图判断，接纳后才计入。',wraplength=370).pack(fill='x',padx=10,pady=8)
  frame=ttk.Frame(win);frame.pack(fill='both',expand=True,padx=10);scroll=ttk.Scrollbar(frame,orient='vertical');self.review_list=tk.Listbox(frame,exportselection=False,yscrollcommand=scroll.set);scroll.config(command=self.review_list.yview);scroll.pack(side='right',fill='y');self.review_list.pack(side='left',fill='both',expand=True)
  self.review_list.bind('<<ListboxSelect>>',lambda e:self.review_preview())
  ttk.Button(win,text='定位并放大选中候选',command=self.zoom_review).pack(fill='x',padx=10,pady=(4,0))
  buttons=ttk.Frame(win);buttons.pack(fill='x',padx=10,pady=8)
  ttk.Button(buttons,text='接纳选中候选',command=self.accept_unpaired).pack(side='left')
  ttk.Button(buttons,text='丢弃选中候选',command=self.discard_unpaired).pack(side='left',padx=5)
  ttk.Button(buttons,text='关闭',command=self.dismiss_review).pack(side='right')
  win.protocol('WM_DELETE_WINDOW',self.dismiss_review);self.update_review()
 def update_review(self):
  if self.review_window is None:return
  self.review_count.set(f'待复核 {len(self.unpaired)}');self.review_list.delete(0,'end');factor=self.factor()
  for i,item in enumerate(self.unpaired,1):
   v=sum(p[2] for p in item['points'])/3
   label=f'{v*factor:.4g} {self.unit.get()}' if factor else f'{v:.3g} 像素'
   reason={'candidate':'自动候选','threshold':'其他阈值','sparse':'邻距较大','neighbor':'无平行邻居','fine':'细小补充'}.get(item.get('review_reason'),'未分类')
   refs=duplicate_references(item,self.unpaired,i-1,self.results)
   t=item.get('threshold');tlabel=f' 阈值{t:g}' if t is not None else ''
   prefix='【可能重复】' if refs else '【待核对】';reference=('相关：'+'、'.join(refs)+'  ') if refs else ''
   low='【小于5像素】' if any(p[2]<5 for p in item['points']) else ''
   fallback='【边界回退】' if boundary_fallback(item,self.runmeta) else ''
   contrast='【局部对比弱】' if weak_local_contrast(item) else ''
   self.review_list.insert('end',f'{prefix}{reference}{low}{fallback}{contrast} 候选 {i}  {reason}{tlabel}  {label}')
  self.preview_unpaired=None;self.render()
 def review_preview(self):
  if self.review_window is None:return
  selected=self.review_list.curselection()
  if not selected:self.preview_unpaired=None;self.render();return
  self.preview_unpaired=self.unpaired[selected[0]];self.render()
  pts=self.preview_unpaired['points'];x=sum((a[0]+b[0])/2 for a,b,_ in pts)/3;y=sum((a[1]+b[1])/2 for a,b,_ in pts)/3
  w=max(self.canvas.winfo_width(),self.im.width*self.zoom);h=max(self.canvas.winfo_height(),self.im.height*self.zoom)
  self.canvas.xview_moveto(max(0,(self.ox+x*self.zoom-self.canvas.winfo_width()/2)/w))
  self.canvas.yview_moveto(max(0,(self.oy+y*self.zoom-self.canvas.winfo_height()/2)/h))
 def zoom_review(self):
  if self.review_window is None or not self.review_list.curselection():self.info.set('请先选择待复核候选。');return
  item=self.unpaired[self.review_list.curselection()[0]];ends=[p for a,b,_ in item['points'] for p in (a,b)]
  span_x=max(p[0] for p in ends)-min(p[0] for p in ends);span_y=max(p[1] for p in ends)-min(p[1] for p in ends)
  width=max(1,self.canvas.winfo_width());height=max(1,self.canvas.winfo_height());base=min(width/self.im.width,height/self.im.height)
  target=min(3.,max(.01,(width-40)/(span_x+100)),max(.01,(height-40)/(span_y+100)))
  self.zoom_factor=min(6.,max(self.zoom_factor,target/base));self.review_preview();self.info.set('已定位并放大候选；放大不会增加原图分辨率。')
 def dismiss_review(self):
  self.close_review();self.render()
 def accept_unpaired(self):
  if self.review_window is None or not self.review_list.curselection():self.info.set('请先选择待复核候选。');return
  idx=self.review_list.curselection()[0];item=self.unpaired[idx]
  if any(p[2]<5 for p in item['points']) and not messagebox.askyesno('测点像素不足','至少一个测点小于 5 像素。细小组织的测量误差可能很大，建议使用更高分辨率原图；请确认边界及测点后再决定。仍要接纳吗？'):return
  refs=duplicate_references(item,[],0,self.results)
  if refs and not messagebox.askyesno('可能重复','该候选可能与 '+ '、'.join(refs)+' 重合。请检查三点位置；仍要接纳吗？'):return
  self.unpaired.pop(idx);item['source']='人工接纳';self.results.append(item);self.refresh();self.update_review();self.info.set(f'已接纳一条；还有 {len(self.unpaired)} 条待复核。')
 def discard_unpaired(self):
  if self.review_window is None or not self.review_list.curselection():self.info.set('请先选择待复核候选。');return
  self.rejected.append(self.unpaired.pop(self.review_list.curselection()[0]));self.update_review();self.info.set(f'已丢弃一条；还有 {len(self.unpaired)} 条待复核。')
 def remove_selected(self):
  selected=self.table.selection()
  if not selected:self.info.set('请先在下方列表中选中片层。');return
  idx=self.table.index(selected[0])
  if 0<=idx<len(self.results):self.results.pop(idx);self.refresh();self.info.set('已删除选中的片层。')
 def setzoom(self,factor):
  if self.im is None:return
  self.zoom_factor=1 if factor==0 else max(.5,min(6,self.zoom_factor*factor));self.render()
 def cancel(self):
  self.pending=[];self.edit_target=None;self.preview_region=None;self.mode='';self.render();self.info.set('已取消当前操作。')
 def setmode(self,m):
  if self.im is None:return
  self.preview_region=None;self.mode=m;self.pending=[];self.info.set({'scale':'请点击标尺两端。','roi':'请点击片层区域矩形的两个对角。','remove':'点击要剔除的片层附近；仅删除最近的候选。','manual':'对同一片层依次点击三条厚度线的两端，共点击六次。','polygon':'沿片层区域边界依次点击顶点，结束后点击“完成多边形选区”。','exclude':'沿要排除的区域点击顶点，再从工具菜单选择“完成排除区”。'}[m]);self.render()
 def finish_polygon(self):
  if self.mode!='polygon' or len(self.pending)<3:
   self.info.set('请先选择至少三个顶点。');return
  self.roi=list(self.pending);self.pending=[];self.mode='';self.render();self.info.set('多边形选区已完成。')
 def finish_exclusion(self):
  if self.mode!='exclude' or len(self.pending)<3:self.info.set('请先选择至少三个排除区顶点。');return
  self.exclusions.append(list(self.pending));self.pending=[];self.mode='';self.render();self.info.set('排除区已添加；重新识别后生效。')
 def clear_exclusions(self):
  self.exclusions=[];self.render();self.info.set('已清空排除区；重新识别后生效。')
 def resetroi(self):self.roi=None;self.render();self.info.set('已取消片层选区；排除区仍保留。重新识别后生效。')
 def click(self,e):
  if self.im is None:return
  x=(self.canvas.canvasx(e.x)-self.ox)/self.zoom;y=(self.canvas.canvasy(e.y)-self.oy)/self.zoom
  if not (0<=x<self.im.width and 0<=y<self.im.height):return
  if not self.mode:
   if self.show_review.get() and self.unpaired:
    distances=[min(point_to_segment(x,y,a,b) for a,b,_ in item['points']) for item in self.unpaired];idx=int(np.argmin(distances))
    if distances[idx]<=7/max(self.zoom,1e-9):
     self.close_regions();self.open_review();self.review_list.selection_clear(0,'end');self.review_list.selection_set(idx);self.review_list.see(idx);self.review_preview();return
   if self.show_regions.get():
    pending=self.pending_regions();hits=[(i,item) for i,item in enumerate(pending) if item['bbox'][0]<=x<item['bbox'][2] and item['bbox'][1]<=y<item['bbox'][3]]
    if hits:
     idx,item=min(hits,key=lambda pair:(pair[1]['bbox'][2]-pair[1]['bbox'][0])*(pair[1]['bbox'][3]-pair[1]['bbox'][1]))
     self.close_review();self.open_regions();self.region_filter.set('全部');self.update_regions();self.region_list.selection_clear(0,'end');self.region_list.selection_set(idx);self.region_list.see(idx);self.region_preview();return
   return
  if self.mode in ('scale','roi'):
   self.pending.append((x,y));self.render()
   if len(self.pending)==2:
    a,b=self.pending;self.pending=[]
    if self.mode=='scale':
     if np.hypot(a[0]-b[0],a[1]-b[1])<2:self.info.set('标尺太短，请重新选择。');return
     self.scalepts=[a,b];self.info.set('已标记标尺两端。请输入该图标尺的实际长度并选择单位。')
    else:
     x0,y0,x1,y1=int(min(a[0],b[0])),int(min(a[1],b[1])),int(max(a[0],b[0])),int(max(a[1],b[1]))
     if x1-x0<3 or y1-y0<3:self.info.set('选区太小，请重新选择两个对角。');self.render();return
     self.roi=(x0,y0,x1,y1);self.info.set('区域已选择，识别只在框内进行。')
    self.mode='';self.refresh()
  elif self.mode in ('polygon','exclude'):
   self.pending.append((x,y));self.render();self.info.set(f'已选择 {len(self.pending)} 个顶点；完成后选择对应的“完成”操作。')
  elif self.mode in ('manual','edit'):
   self.pending.append((x,y));self.render()
   if len(self.pending)==6:
    pts=[]
    for j in (0,2,4):
     a,b=self.pending[j:j+2];pts.append((a,b,float(np.hypot(a[0]-b[0],a[1]-b[1]))))
    if min(p[2] for p in pts)<1:self.info.set('测点距离过短，请重新选择。');self.pending=[];return
    if repeated_measurement_positions(pts):
     self.pending=[];self.render();self.info.set('测点位置重复，请在同一片层的三个不同位置重新画线。');messagebox.showwarning('测点位置重复','三个截距中有两个位置几乎重合，无法作为三个不同位置的测量。请重新选择。');return
    region=self.runmeta.get('roi') if self.runmeta else self.roi;exclusions=self.runmeta.get('exclusions',[]) if self.runmeta else self.exclusions
    mask=self.roi_from(region,exclusions)
    if any(not segment_inside_roi(a,b,mask) for a,b,_ in pts) and not messagebox.askyesno('测线越过分析范围','至少一条测量线位于分析范围外或穿过排除区。请核对是否为需要统计的片层；仍要保存吗？'):
     self.pending=[];self.render();self.info.set('未保存这些测点，请重新选择；原有测量保持不变。');return
    if inconsistent_manual_directions(pts) and not messagebox.askyesno('测点方向差异较大','三条厚度线的方向相差超过 45°。请确认它们位于同一片层，并尽量垂直于各自位置的片层边界。弯曲片层也可能出现此提示；仍要保存吗？'):
     self.pending=[];self.render();self.info.set('未保存这些测点，请重新选择；原有测量保持不变。');return
    if min(p[2] for p in pts)<5 and not messagebox.askyesno('测点像素不足','至少一个手动测点小于 5 像素，测量误差可能很大。建议使用更高分辨率原图；仍要保存这三个测点吗？'):
     self.pending=[];self.render();self.info.set('未保存这些测点，请重新选择；原有测量保持不变。');return
    if self.mode=='edit':
     if not any(x is self.edit_target for x in self.results):self.pending=[];self.mode='';self.info.set('原片层已不存在，请重新选择。');return
     self.edit_target['points']=pts;self.edit_target['source']='人工修正';self.edit_target['coherence']=None;self.edit_target['contrast_score']=None;self.edit_target['contour']=[];self.edit_target['edge_methods']=None;self.edit_target['neighbor_count']=None;self.edit_target['compact_neighbor_count']=None;self.edit_target['threshold']=None;self.edit_target['review_reason']=None;self.pending=[];self.mode='';self.edit_target=None;self.refresh();self.info.set('已修正选中片层的三个测点。')
    else:
     self.results.append({'points':pts,'length':0,'bbox':None,'source':'手动'});self.runmeta=self.runmeta or {'roi':self.roi,'exclusions':[list(poly) for poly in self.exclusions],'phase':'手动','params':{}};self.pending=[];self.preview_region=None;self.refresh();self.info.set('已补充一个片层；待辨区域需单独标记已检查。')
   else:self.info.set(f'已选择 {len(self.pending)}/6 个端点。')
  elif self.mode=='remove' and self.results:
   ds=[min(np.hypot(x-(a[0]+b[0])/2,y-(a[1]+b[1])/2) for a,b,_ in item['points']) for item in self.results];idx=int(np.argmin(ds))
   if ds[idx]<25/self.zoom:self.results.pop(idx);self.refresh();self.info.set('已删除候选；可继续点击删除。')
 def factor(self):
  try:
   v=float(self.scale.get());a,b=self.scalepts
   if not np.isfinite(v) or v<=0 or self.unit.get() not in ('µm','nm','mm'): return None
   pixels=float(np.hypot(a[0]-b[0],a[1]-b[1]))
   if not np.isfinite(pixels) or pixels<2:return None
   value=v/pixels
   if not np.isfinite(value) or value<=0:return None
   if self.im is not None and not np.isfinite(value*float(np.hypot(self.im.width,self.im.height))):return None
   return value
  except (ValueError,TypeError,OverflowError,ZeroDivisionError):return None
 def run(self):
  if self.im is None:return
  if self.factor() is None:messagebox.showwarning('需要校准','请标记相距至少 2 像素的标尺两端，输入合理的正数长度并选择单位；过大或过小的数值无法计算。');return
  try:
   vals=[float(v.get()) for v in self.entries.values()];mi,ma,length,ratio,sens,coh=vals
   if not all(np.isfinite(vals)) or not 0<mi<ma or length<=0 or ratio<1 or not 0<=sens<=255 or not 0<=coh<=1:raise ValueError('请检查厚度、长度、长宽比和阈值。')
  except ValueError as e:messagebox.showerror('参数错误',str(e));return
  if (self.results or self.unpaired or self.uncertain_regions) and not messagebox.askyesno('重新识别','重新识别会替换当前自动和手动测量。请先保存工程；继续吗？'):return
  roi=self.analysis_roi()
  self.info.set('正在分析，请稍候……');self.root.update_idletasks()
  try:
   raw=detect(self.im,roi,'dark' if self.phase.get()=='暗相' else 'bright',mi,ma,length,ratio,sens,coherence_min=coh,boundary_method=self.boundary.get(),neighbor_width_factor=self.neighbor_factor)
   regions=find_uncertain_regions(self.im,roi,'dark' if self.phase.get()=='暗相' else 'bright',mi,ma,length,sens,raw,coherence_min=coh)
   for region in regions:region['state']='待检查'
   regions.extend(review_tiles(roi))
   for item in raw:
    item['review_reason']='candidate' if item['compact_neighbor_count']>0 else ('sparse' if item['neighbor_count']>0 else 'neighbor')
   unpaired=sorted(raw,key=lambda item:item['compact_neighbor_count']==0) if self.parallel_only.get() else raw
   extra_threshold=None;extra_count=0
   if self.auto_extra.get():
    extra_threshold=12. if abs(sens-12.)>1e-9 else 1.
    extra=detect(self.im,roi,'dark' if self.phase.get()=='暗相' else 'bright',mi,ma,length,ratio,extra_threshold,coherence_min=coh,boundary_method=self.boundary.get(),neighbor_width_factor=self.neighbor_factor)
    extra_regions=find_uncertain_regions(self.im,roi,'dark' if self.phase.get()=='暗相' else 'bright',mi,ma,length,extra_threshold,extra,coherence_min=coh)
    for region in extra_regions:region['state']='待检查'
    regions.extend(extra_regions)
    for item in extra:item['review_reason']='threshold'
    extra.sort(key=lambda item:any(likely_same_lamella(item,existing) for existing in unpaired))
    extra_count=len(extra);unpaired=unpaired+extra
   fine_count=0;fine_region_count=0;fine_params=None
   if self.auto_fine.get() and not unpaired and (mi,ma,length,ratio,coh)==(2.5,20.,30.,4.,.7):
    fine_params={'min_width':1.,'max_width':8.,'min_length':20.,'elongation':3.,'coherence_min':.5}
    self.info.set('默认参数无候选，正在补充细小组织候选；低像素结果须审核。');self.root.update_idletasks()
    for t in ([sens,extra_threshold] if extra_threshold is not None else [sens]):
     fine=detect(self.im,roi,'dark' if self.phase.get()=='暗相' else 'bright',1.,8.,20.,3.,t,coherence_min=.5,boundary_method=self.boundary.get(),neighbor_width_factor=self.neighbor_factor)
     fine_regions=find_uncertain_regions(self.im,roi,'dark' if self.phase.get()=='暗相' else 'bright',1.,8.,20.,t,fine,coherence_min=.5)
     for region in fine_regions:region['state']='待检查';region['analysis_mode']='fine'
     fine_region_count+=len(fine_regions);regions.extend(fine_regions)
     for item in fine:item['review_reason']='fine'
     fine_count+=len(fine);unpaired.extend(fine)
   accepted=[]
  except Exception as e:messagebox.showerror('识别失败',str(e));return
  self.close_review();self.close_regions();self.uncertain_regions=regions;self.results=accepted;self.rejected=[];self.unpaired=unpaired;rejected=sum(x['compact_neighbor_count']==0 for x in unpaired)
  self.runmeta={'roi':self.roi,'exclusions':[list(poly) for poly in self.exclusions],'phase':self.phase.get(),'boundary':self.boundary.get(),'parallel_only':self.parallel_only.get(),'neighbor_factor':self.neighbor_factor,'neighbor_rejected':rejected,'initial_review_count':len(unpaired),'initial_uncertain_regions':len(regions),'auto_extra':self.auto_extra.get(),'extra_threshold':extra_threshold,'extra_count':extra_count,'auto_fine':self.auto_fine.get(),'fine_count':fine_count,'fine_region_count':fine_region_count,'fine_params':fine_params,'params':{k:v.get() for k,v in self.entries.items()}};self.mode='';self.refresh()
  self.info.set((f'默认参数无候选，补充 {fine_count} 条细小候选和 {fine_region_count} 个待辨区域（低像素需复核）；' if fine_params else '')+(f'{len(self.unpaired)} 条候选、{len(self.uncertain_regions)} 个待辨区域均未计入统计；未标区域也须检查。' if (self.unpaired or self.uncertain_regions) else '未发现候选，不代表没有片层；请检查图片，调整参数或手动补充。'))
 def render(self):
  if self.im is None:return
  w=max(1,self.canvas.winfo_width());h=max(1,self.canvas.winfo_height());self.zoom=min(w/self.im.width,h/self.im.height)*self.zoom_factor;self.ox=max(0,(w-self.im.width*self.zoom)/2);self.oy=max(0,(h-self.im.height*self.zoom)/2);self.canvas.configure(scrollregion=(0,0,max(w,self.im.width*self.zoom),max(h,self.im.height*self.zoom)))
  resample=Image.Resampling.NEAREST if self.zoom>=1 else Image.Resampling.LANCZOS
  img=self.preview_image(self.display_mode.get()).resize((max(1,int(self.im.width*self.zoom)),max(1,int(self.im.height*self.zoom))),resample=resample);self.photo=ImageTk.PhotoImage(img);c=self.canvas;c.delete('all');c.create_image(self.ox,self.oy,anchor='nw',image=self.photo)
  def conv(p):return (self.ox+p[0]*self.zoom,self.oy+p[1]*self.zoom)
  if self.mode in ('polygon','exclude') and len(self.pending)>1:c.create_line(*[z for point in self.pending for z in conv(point)],fill='cyan',width=2)
  for p in self.pending:
   xx,yy=conv(p);c.create_oval(xx-3,yy-3,xx+3,yy+3,fill='cyan',outline='black')
  if self.roi:
   if isinstance(self.roi,list):c.create_polygon(*[z for point in self.roi for z in conv(point)],outline='#00eecc',fill='',width=2)
   else:
    x0,y0,x1,y1=self.roi;c.create_rectangle(*conv((x0,y0)),*conv((x1,y1)),outline='#00eecc',width=2)
  for polygon in self.exclusions:
   c.create_polygon(*[v for p in polygon for v in conv(p)],outline='#ff9a27',fill='',width=2)
  if len(self.scalepts)==2:c.create_line(*conv(self.scalepts[0]),*conv(self.scalepts[1]),fill='cyan',width=2)
  selection=self.table.selection() if hasattr(self,'table') else ()
  selected=self.table.index(selection[0]) if selection else -1
  if self.show_review.get() and self.runmeta and self.runmeta.get('phase') in ('暗相','亮相'):
   region=self.runmeta.get('roi')
   if not region:region=(0,0,self.im.width-1,int(self.im.height*.91)-1)
   if len(region)==4 and all(isinstance(v,(int,float)) for v in region):
    x0,y0,x1,y1=region;c.create_rectangle(*conv((x0,y0)),*conv((x1,y1)),outline='#c878ff',width=2,dash=(5,3),tags=('review_scope',))
   else:c.create_polygon(*[v for p in region for v in conv(p)],outline='#c878ff',fill='',width=2,dash=(5,3),tags=('review_scope',))
   c.create_text(*conv((4,4)),text='紫框内未标区域仍需人工检查',anchor='nw',fill='#e3b6ff',font=('Microsoft YaHei',9,'bold'),tags=('review_scope',))
  if self.show_regions.get():
   for index,item in enumerate(self.pending_regions(),1):
    x0,y0,x1,y1=item['bbox'];tile=item.get('kind')=='tile';color='#cf92ff' if tile else '#42d8ff'
    if self.display_mode.get()=='伪彩增强':c.create_rectangle(*conv((x0,y0)),*conv((x1,y1)),outline='#101010',width=3,dash=(3,2),tags=('region_underlay',))
    c.create_rectangle(*conv((x0,y0)),*conv((x1,y1)),outline=color,width=1,dash=(3,2),tags=('uncertain_region',));c.create_text(*conv((x0,y0)),text=f'{"块" if tile else "区"}{index}',anchor='nw',fill=color,font=('Microsoft YaHei',8),tags=('uncertain_region_label',))
  if self.preview_region:
   x0,y0,x1,y1=self.preview_region['bbox'];c.create_rectangle(*conv((x0,y0)),*conv((x1,y1)),outline='#ffef4a',width=3,tags=('uncertain_region_preview',))
  if self.show_review.get():
   for index,item in enumerate(self.unpaired,1):
    for point,(a,b,px) in enumerate(item['points'],1):
     if self.display_mode.get()=='伪彩增强':c.create_line(*conv(a),*conv(b),fill='#101010',width=5,tags=('candidate_underlay',))
     c.create_line(*conv(a),*conv(b),fill='#ff9a27',width=2,tags=('candidate_line',f'candidate_{index}'))
     if self.factor():
      label=f'{index}.{point}:{px*self.factor():.3g}';x,y=conv(b)
      if self.display_mode.get()=='伪彩增强':c.create_text(x+1,y+1,text=label,anchor='sw',fill='black',font=('Arial',7),tags=('candidate_label_underlay',))
      c.create_text(x,y,text=label,anchor='sw',fill='#ffbe73',font=('Arial',7),tags=('candidate_measurement',f'candidate_{index}'))
    c.create_text(*conv(item['points'][1][0]),text=f'待{index}',anchor='sw',fill='#ffd0a0',font=('Microsoft YaHei',8,'bold'))
  if self.preview_unpaired:
   outline=self.preview_unpaired.get('contour',[])
   if len(outline)>1:c.create_line(*[v for p in outline for v in conv(p)],fill='#ff9a27',width=2)
   for point,(a,b,px) in enumerate(self.preview_unpaired['points'],1):
    c.create_line(*conv(a),*conv(b),fill='#ffef4a',width=3)
    if self.factor():c.create_text(*conv(b),text=f'{point}: {px*self.factor():.4g} {self.unit.get()}',anchor='sw',fill='#fff56a',font=('Arial',9,'bold'))
   c.create_text(*conv(self.preview_unpaired['points'][1][0]),text='待复核',anchor='sw',fill='#ffba54',font=('Microsoft YaHei',10,'bold'))
  for i,item in enumerate(self.results,1):
   if i-1==selected and len(item.get('contour',[]))>1:c.create_line(*[v for p in item['contour'] for v in conv(p)],fill='#00e9fa',width=2)
   for j,(a,b,px) in enumerate(item['points'],1):
    c.create_line(*conv(a),*conv(b),fill='#2bff66' if i-1==selected else '#ff3a34',width=3 if i-1==selected else 2)
    if self.factor():c.create_text(*conv(b),text=f'{i}.{j}:{px*self.factor():.4g}',anchor='sw',fill='#fff38a',font=('Arial',7))
   a,b,_=item['points'][1];c.create_text(*conv(a),text=str(i),anchor='sw',fill='yellow',font=('Arial',9,'bold'))
 def refresh(self):
  self.review_count.set(f'待复核 {len(self.unpaired)}');self.region_count.set(f'待辨区域 {len(self.pending_regions())}')
  for i in self.table.get_children():self.table.delete(i)
  f=self.factor();scope=self.roi_from(self.runmeta.get('roi'),self.runmeta.get('exclusions',[])) if self.im is not None and self.runmeta and self.results else None
  for i,item in enumerate(self.results,1):
   v=[p[2]*f for p in item['points']] if f else None;flags=[]
   if any(p[2]<5 for p in item['points']):flags.append('有测点<5像素')
   if repeated_measurement_positions(item['points']):flags.append('测点位置重复，不计统计')
   if weak_local_contrast(item):flags.append('局部对比弱')
   if scope is not None and measurement_outside(item,scope):flags.append('测线越界或进入排除区')
   if self.runmeta and self.runmeta.get('boundary')=='half' and item.get('edge_methods') and any(v=='mask' for pair in item['edge_methods'] for v in pair):flags.append('边界回退')
   if item.get('neighbor_count')==0:flags.append('无平行邻居')
   elif item.get('compact_neighbor_count')==0:flags.append('邻距较大')
   numbers=[f'{x:.4g}' for x in v] if v else ['—']*3
   mean=f'{sum(v)/3:.4g} {self.unit.get()}' if v else '未校准'
   self.table.insert('','end',values=(i,*numbers,mean,'；'.join(flags)))
  if f and self.results:
   valid=[item for item in self.results if not repeated_measurement_positions(item['points'])];excluded=len(self.results)-len(valid)
   means=np.array([sum(p[2] for p in item['points'])/3*f for item in valid]);sd=float(np.std(means,ddof=1)) if len(means)>1 else None
   low=sum(any(p[2]<5 for p in item['points']) for item in valid);mean=f'{np.mean(means):.4g} {self.unit.get()}' if len(means) else '—'
   self.summary.set(f'有效片层数：{len(means)}   平均：{mean}\n片层平均值的样本标准差：'+(f'{sd:.4g}' if sd is not None else '—')+f'\n至少一处小于 5 像素：{low} 条；重复测点不计统计：{excluded} 条')
  else:self.summary.set('请完成标尺校准' if self.results else (f'已确认：0 条；待审核 {len(self.unpaired)} 条（不计入统计）' if self.unpaired else '尚无统计结果'))
  self.render()
 def export_review(self):
  if self.im is None or not self.runmeta or self.runmeta.get('phase') not in ('暗相','亮相'):messagebox.showwarning('尚未识别','请先完成一次自动识别。');return
  path=filedialog.asksaveasfilename(defaultextension='.png',filetypes=[('PNG 图片','*.png')])
  if not path:return
  if os.path.splitext(path)[1].lower()!='.png':messagebox.showerror('无法导出','请使用 .png 文件名保存待审核标注图。');return
  csvpath=os.path.splitext(path)[0]+'_候选.csv';regionpath=os.path.splitext(path)[0]+'_待辨区域.csv';cardpath=os.path.splitext(path)[0]+'_逐条审核.html'
  if any(same_output_file(target,self.path) for target in (path,csvpath,regionpath,cardpath)):
   messagebox.showerror('无法导出','导出文件会覆盖原始图片，请选择其他文件名。');return
  existing=[p for p in (csvpath,regionpath,cardpath) if os.path.exists(p)]
  if existing and not messagebox.askyesno('覆盖已有文件','以下配套文件已存在：\n'+'\n'.join(os.path.basename(p) for p in existing)+'\n确定覆盖吗？'):return
  factor=self.factor();unit=self.unit.get() if factor else '像素（未校准）';k=factor if factor else 1.
  try:
   # Keep every source-image pixel visible: put the legend above the image.
   offset=54;img=Image.new('RGB',(self.im.width,self.im.height+offset),'#163349');img.paste(self.im,(0,offset));draw=ImageDraw.Draw(img)
   try:font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
   except OSError:font=ImageFont.load_default()
   draw.text((5,3),f'橙线：待审核测点；蓝框：待辨区域（无厚度）；单位：{unit}',font=font,fill='white')
   draw.text((5,20),'紫框内未标区域仍需人工检查；所有待审核项均未计入统计。',font=font,fill='white')
   draw.text((5,37),'图片向下偏移 54 像素；CSV 坐标以原图为准。',font=font,fill='white')
   region=self.runmeta.get('roi')
   if not region:region=(0,0,self.im.width-1,int(self.im.height*.91)-1)
   if len(region)==4 and all(isinstance(v,(int,float)) for v in region):
    x0,y0,x1,y1=region;draw.rectangle((x0,y0+offset,x1,y1+offset),outline='#c878ff',width=2)
   else:
    coords=[(p[0],p[1]+offset) for p in region];draw.line(coords+[coords[0]],fill='#c878ff',width=2)
   for poly in self.runmeta.get('exclusions',[]):
    if len(poly)>=3:
     coords=[(p[0],p[1]+offset) for p in poly];draw.line(coords+[coords[0]],fill='#b0b0b0',width=2)
   for i,item in enumerate(self.pending_regions(),1):
    x0,y0,x1,y1=item['bbox'];color='#cf92ff' if item.get('kind')=='tile' else '#42d8ff';draw.rectangle((x0,y0+offset,x1,y1+offset),outline=color,width=1);draw.text((x0,y0+offset),f'R{i}',font=font,fill=color,stroke_width=1,stroke_fill='black')
   for i,item in enumerate(self.unpaired,1):
    for j,(a,b,px) in enumerate(item['points'],1):
     aa=(a[0],a[1]+offset);bb=(b[0],b[1]+offset)
     draw.line((aa,bb),fill='#ff9a27',width=2);draw.text(bb,f'{i}.{j}:{px*k:.3g}',font=font,fill='#ffe599',stroke_width=1,stroke_fill='black')
   img.save(path)
   with open(csvpath,'w',encoding='utf-8-sig',newline='') as out:
    wr=csv.writer(out);wr.writerow(['状态','候选编号','来源','阈值','可能与已确认或前序候选重复','点1','点2','点3','三点均值','单位','点1起点X','点1起点Y','点1终点X','点1终点Y','点2起点X','点2起点Y','点2终点X','点2终点Y','点3起点X','点3起点Y','点3终点X','点3终点Y','边界回退提示','点1边界','点2边界','点3边界','疑似重复对象','局部对比指标（灰度级）','局部对比弱'])
    for i,item in enumerate(self.unpaired,1):
     values=[p[2]*k for p in item['points']];refs=duplicate_references(item,self.unpaired,i-1,self.results)
     reason={'candidate':'自动候选','neighbor':'无平行邻居','sparse':'邻距较大','threshold':'其他阈值','fine':'细小补充'}.get(item.get('review_reason'),'未记录')
     methods=item.get('edge_methods');method_labels=[('/'.join('半高' if v=='half' else '轮廓' for v in pair)) for pair in methods] if methods else ['无记录']*3
     wr.writerow(['待审核',i,reason,item.get('threshold') if item.get('threshold') is not None else '', '可能' if refs else '',*values,sum(values)/3,unit,*[float(v) for a,b,_ in item['points'] for v in (*a,*b)],'是' if boundary_fallback(item,self.runmeta) else '',*method_labels,'、'.join(refs),item.get('contrast_score') if item.get('contrast_score') is not None else '','是' if weak_local_contrast(item) else ''])
   with open(regionpath,'w',encoding='utf-8-sig',newline='') as out:
    wr=csv.writer(out);wr.writerow(['记录编号','当前显示编号','状态','左上X','左上Y','右下X（不含）','右下Y（不含）','提取阈值','局部方向一致性','说明'])
    visible=0
    for index,item in enumerate(self.uncertain_regions,1):
     state=item.get('state','待检查');visible+=int(state=='待检查');number=visible if state=='待检查' else ''
     wr.writerow([index,number,state,*item['bbox'],item['threshold'],item['coherence'],'全范围分块，需人工检查' if item.get('kind')=='tile' else ('细小补充区域，未自动测厚' if item.get('analysis_mode')=='fine' else '仅为可疑区域，未自动测厚')])
   self.export_cards(cardpath,factor,items=self.unpaired,pending=True)
  except Exception as e:messagebox.showerror('导出失败',str(e));return
  self.info.set(f'已导出 {len(self.unpaired)} 条待审核候选和 {len(self.pending_regions())} 个待辨区域，均未计入统计。')
 def export(self):
  f=self.factor()
  if not self.results or not f:messagebox.showwarning('无结果','请先校准并识别片层。');return
  path=filedialog.asksaveasfilename(defaultextension='.csv',filetypes=[('CSV 文件','*.csv')])
  if not path:return
  if os.path.splitext(path)[1].lower()!='.csv':messagebox.showerror('无法导出','请使用 .csv 文件名保存测量结果。');return
  base=os.path.splitext(path)[0];imagepath=base+'_标注.png';cardpath=base+'_测量卡片.html'
  if any(same_output_file(target,self.path) for target in (path,imagepath,cardpath)):
   messagebox.showerror('无法导出','导出文件会覆盖原始图片，请选择其他文件名。');return
  existing=[p for p in (imagepath,cardpath) if os.path.exists(p)]
  if existing and not messagebox.askyesno('覆盖已有文件','以下配套文件已存在：\n'+'\n'.join(os.path.basename(p) for p in existing)+'\n确定覆盖吗？'):return
  img=self.im.copy();draw=ImageDraw.Draw(img)
  # Draw the analysis-region snapshot used for these results, not un-applied edits.
  region=self.runmeta.get('roi')
  if region:
   if len(region)==4 and all(isinstance(v,(int,float)) for v in region):draw.rectangle(region,outline='#00e9fa',width=2)
   else:draw.line([tuple(p) for p in region]+[tuple(region[0])],fill='#00e9fa',width=2)
  for poly in self.runmeta.get('exclusions',[]):
   if len(poly)>=3:draw.line([tuple(p) for p in poly]+[tuple(poly[0])],fill='#ff9a27',width=2)
  for i,item in enumerate(self.results,1):
   for j,(a,b,px) in enumerate(item['points'],1):
    draw.line([a,b],fill='red',width=2);draw.text(b,f'{i}.{j}:{px*f:.4g}',fill='yellow')
   draw.text(item['points'][1][0],str(i),fill='yellow')
  if len(self.scalepts)==2:draw.line(self.scalepts,fill='cyan',width=2)
  img.save(imagepath)
  scope=self.roi_from(self.runmeta.get('roi'),self.runmeta.get('exclusions',[]))
  with open(path,'w',encoding='utf-8-sig',newline='') as out:
   wr=csv.writer(out);wr.writerow(['图片',self.path]);wr.writerow(['标尺实际长度',self.scale.get(),self.unit.get(),'标尺像素',float(self.scale.get())/f]);wr.writerow(['选区',self.runmeta['roi'],'排除区',self.runmeta.get('exclusions',[]),'相别',self.runmeta['phase'],'平行邻居优先排序',('开启' if self.runmeta.get('parallel_only',False) else '关闭') if any(x.get('source','自动') in ('自动','人工接纳') for x in self.results) else '无自动候选','邻距倍宽系数',self.runmeta.get('neighbor_factor','旧版未记录'),'初始待审核数',self.runmeta.get('initial_review_count','旧版未记录'),'自动补充阈值',self.runmeta.get('extra_threshold',''),'补充候选数',self.runmeta.get('extra_count',''),'细小补充数量',self.runmeta.get('fine_count',0),'细小补充区域数',self.runmeta.get('fine_region_count',0),'细小补充参数',self.runmeta.get('fine_params') or '','无紧邻候选数',self.runmeta.get('neighbor_rejected',''),'待复核剩余数',len(self.unpaired),'待辨区域剩余数',len(self.pending_regions()),'自动边界方法',({'half':'局部半高','mask':'分割轮廓'}.get(self.runmeta.get('boundary','mask'),'分割轮廓') if any(x.get('source','自动') in ('自动','人工接纳') for x in self.results) else '无（全部手动或修正）')]);wr.writerow(['参数']+sum(([k,v] for k,v in self.runmeta['params'].items()),[]));means=[sum(p[2] for p in item['points'])/3*f for item in self.results if not repeated_measurement_positions(item['points'])];wr.writerow(['片层数',len(means),'片层平均值的均值',np.mean(means) if means else '', '片层平均值的样本标准差',np.std(means,ddof=1) if len(means)>1 else '', '重复测点不计统计',len(self.results)-len(means)]);wr.writerow(['片层编号','来源','方向一致性','有测点小于5像素','点1边界','点2边界','点3边界','点1','点2','点3','三点平均','单位','点1起点','点1终点','点2起点','点2终点','点3起点','点3终点','候选分割阈值','待复核来源','识别时紧邻候选数','识别时邻近平行候选数','测线越过分析范围或排除区','测点位置重复','是否计入统计','局部对比指标（灰度级）','局部对比弱'])
   for i,item in enumerate(self.results,1):
    v=[p[2]*f for p in item['points']];wr.writerow([i,item.get('source','自动'),item.get('coherence') if item.get('coherence') is not None else '', '是' if any(p[2]<5 for p in item['points']) else '否',*([('/'.join('半高' if x=='half' else '轮廓' for x in pair)) for pair in item['edge_methods']] if item.get('edge_methods') else ['人工或无记录']*3),*v,sum(v)/3,self.unit.get(),*[str(tuple(float(v) for v in q)) for a,b,_ in item['points'] for q in (a,b)],item.get('threshold') if item.get('threshold') is not None else '',{'candidate':'自动候选','neighbor':'无平行邻居','sparse':'邻距较大','threshold':'其他阈值','fine':'细小补充'}.get(item.get('review_reason'),''),item.get('compact_neighbor_count') if item.get('compact_neighbor_count') is not None else '',item.get('neighbor_count') if item.get('neighbor_count') is not None else '', '是' if measurement_outside(item,scope) else '否','是' if repeated_measurement_positions(item['points']) else '否','否' if repeated_measurement_positions(item['points']) else '是',item.get('contrast_score') if item.get('contrast_score') is not None else '','是' if weak_local_contrast(item) else ''])
  self.export_cards(cardpath,f)
  self.info.set('已保存 CSV、标注图和逐片层测量卡片。')
 def export_cards(self,path,factor,items=None,pending=False):
  cards=[];items=self.results if items is None else items;k=factor if factor else 1.;result_scope=self.roi_from(self.runmeta.get('roi'),self.runmeta.get('exclusions',[])) if not pending and self.runmeta else None
  for i,item in enumerate(items,1):
   points=item['points'];coords=[q for a,b,_ in points for q in (a,b)]
   x0=max(0,int(min(p[0] for p in coords))-25);y0=max(0,int(min(p[1] for p in coords))-25)
   x1=min(self.im.width,int(max(p[0] for p in coords))+26);y1=min(self.im.height,int(max(p[1] for p in coords))+26)
   crop=self.im.crop((x0,y0,x1,y1));scale=min(4.,max(1.,600/max(crop.size)))
   crop=crop.resize((max(1,int(crop.width*scale)),max(1,int(crop.height*scale))),Image.Resampling.NEAREST)
   d=ImageDraw.Draw(crop)
   contour=item.get('contour',[])
   if len(contour)>1:d.line([((x-x0)*scale,(y-y0)*scale) for x,y in contour],fill='cyan',width=max(1,int(scale)))
   for j,(a,b,px) in enumerate(points,1):
    aa=((a[0]-x0)*scale,(a[1]-y0)*scale);bb=((b[0]-x0)*scale,(b[1]-y0)*scale)
    d.line([aa,bb],fill='#ff9a27' if pending else 'red',width=max(2,int(scale)));d.text(bb,str(j),fill='yellow',stroke_width=1,stroke_fill='black')
   buf=io.BytesIO();crop.save(buf,format='PNG');src=base64.b64encode(buf.getvalue()).decode('ascii')
   vals=[p[2]*k for p in points];unit=html.escape(self.unit.get() if factor else '像素（未校准）');kind='待审核' if pending else html.escape(item.get('source','自动'))
   methods=item.get('edge_methods')
   desc='　'.join(f'点{j}: {v:.5g} {unit}'+('（'+'/'.join('半高' if x=='half' else '轮廓' for x in methods[j-1])+'）' if methods else '（人工或无记录）') for j,v in enumerate(vals,1))
   threshold_label=f"候选阈值：{item['threshold']:g}" if item.get('threshold') is not None else '候选阈值：人工或无记录'
   if item.get('review_reason')=='fine':threshold_label+='；细小补充候选（宽度 1–8 像素，最短长度 20 像素，长宽比 3，方向一致性 0.5）'
   warning='有测点小于 5 像素，请重点核对' if any(p[2]<5 for p in points) else ''
   if weak_local_contrast(item):warning+='；局部对比偏弱，请核对相界和噪声'
   if repeated_measurement_positions(points):warning+='；测点位置重复，不计统计，请重新测量'
   if result_scope is not None and measurement_outside(item,result_scope):warning+='；测线越过分析范围或进入排除区'
   if boundary_fallback(item,self.runmeta):warning+='；部分边界由半高回退到分割轮廓，请重点核对'
   refs=duplicate_references(item,items,i-1,self.results) if pending else []
   if refs:warning+='；可能与已确认或前序候选重复，请核对；相关：'+'、'.join(refs)
   context_html=''
   if pending:
    left=max(0,x0-100);top=max(0,y0-100);right=min(self.im.width,x1+100);bottom=min(self.im.height,y1+100)
    context=self.im.crop((left,top,right,bottom)).convert('RGB');dctx=ImageDraw.Draw(context);dctx.rectangle((x0-left,y0-top,x1-left-1,y1-top-1),outline='#ffff00',width=2)
    size=min(1.,600/max(context.size));context=context.resize((max(1,int(context.width*size)),max(1,int(context.height*size))),Image.Resampling.LANCZOS)
    ctxbuf=io.BytesIO();context.save(ctxbuf,format='JPEG',quality=85);ctx=base64.b64encode(ctxbuf.getvalue()).decode('ascii')
    context_html=f'<p>周边原图（黄框为上方放大范围；请检查是否靠近粗大组织）</p><img src="data:image/jpeg;base64,{ctx}" alt="候选 {i} 周边原图">'
   cards.append(f'<article data-kind="candidate"><h2>片层 {i} · {kind}</h2><p>{threshold_label}</p><p>{warning}</p><img src="data:image/png;base64,{src}" alt="片层 {i} 的三个测点"><p>{desc}</p><p>三点平均：{sum(vals)/3:.5g} {unit}</p>{context_html}</article>')
  if pending:
   # Region rectangles can cross polygon boundaries or exclusions. Dim pixels
   # outside the original analysis mask so exported cards preserve that context.
   scope=self.roi_from(self.runmeta.get('roi'),self.runmeta.get('exclusions',[]))
   scope_image=Image.fromarray((scope.astype('uint8')*255),mode='L')
   for i,region in enumerate(self.pending_regions(),1):
    x0,y0,x1,y1=region['bbox'];pad=16;left=max(0,x0-pad);top=max(0,y0-pad);right=min(self.im.width,x1+pad);bottom=min(self.im.height,y1+pad)
    crop=self.im.crop((left,top,right,bottom)).convert('RGB');mask=scope_image.crop((left,top,right,bottom));dim=crop.point(lambda value: int(value*.35));crop=Image.composite(crop,dim,mask);scale=min(3.,600/max(crop.size));crop=crop.resize((max(1,int(crop.width*scale)),max(1,int(crop.height*scale))),Image.Resampling.NEAREST)
    d=ImageDraw.Draw(crop);d.rectangle(((x0-left)*scale,(y0-top)*scale,(x1-left)*scale,(y1-top)*scale),outline='#42d8ff',width=max(1,int(scale)))
    buf=io.BytesIO();crop.save(buf,format='PNG');src=base64.b64encode(buf.getvalue()).decode('ascii')
    label='全范围分块' if region.get('kind')=='tile' else f"{'细小补充区域' if region.get('analysis_mode')=='fine' else '可疑区域'}（阈值 {region['threshold']:g}）"
    cards.append(f'<article data-kind="region"><h2>{label} R{i}（无自动厚度）</h2><img src="data:image/png;base64,{src}" alt="{label} R{i}"><p>原图范围：({x0}, {y0})–({x1}, {y1})；变暗部分在分析范围外或排除区内。可能包含粗大组织，需人工检查；此区域未计入厚度统计。</p></article>')
  title='待审核候选与待辨区域（未计入统计）' if pending else '已确认片层三点测量卡片'
  page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>片层测量卡片</title><style>body{font-family:Microsoft YaHei,sans-serif;background:#eef2f3;color:#19333b;margin:24px} main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}article{background:white;padding:16px;border-radius:8px;break-inside:avoid}h2{font-size:17px}img{max-width:100%;image-rendering:pixelated}p{line-height:1.5} @media print{body{background:white}main{display:block}article{page-break-inside:avoid}}</style><h1>'+title+'</h1><p>图片：'+html.escape(os.path.basename(self.path))+'。橙线表示待审核测点，红线表示已确认测点；黄色数字对应点号。候选卡片中的青线为算法分割轮廓；区域卡片中的蓝框是无法自动测厚的待辨区域，不代表片层。半高边界测点可与分割轮廓不重合；划痕、成像条纹和噪声也可能形成候选，未标出的区域同样须人工检查。</p><main>'+''.join(cards)+'</main></html>'
  with open(path,'w',encoding='utf-8') as out:out.write(page)
if __name__=='__main__':
 root=tk.Tk();App(root);root.mainloop()
