import numpy as np
from scipy import ndimage as ndi
from skimage import morphology, measure


def enhance_sem_display(image, color=True):
    """Contrast-limited preview only; never used to calculate measurements."""
    from skimage import exposure
    from PIL import Image
    gray=np.asarray(image.convert('L'),dtype=np.uint8)
    if gray.min()==gray.max():enhanced=gray
    elif min(gray.shape)<8:enhanced=exposure.rescale_intensity(gray,out_range='uint8').astype(np.uint8)
    else:enhanced=np.rint(exposure.equalize_adapthist(gray,clip_limit=0.01)*255).clip(0,255).astype(np.uint8)
    if not color:return Image.fromarray(enhanced).convert('RGB')
    stops=np.array([[12,17,53],[28,79,112],[36,146,151],[177,202,150],[255,239,193]],dtype=np.float32)
    index=np.linspace(0,255,len(stops));lut=np.stack([np.interp(np.arange(256),index,stops[:,i]) for i in range(3)],axis=1).astype(np.uint8)
    return Image.fromarray(lut[enhanced],mode='RGB')


def suggest_contrast_threshold(image, roi=None):
    """Empirical starting point, not a calibration of SEM imaging noise."""
    gray=np.asarray(image.convert('L'),dtype=np.float32)
    if roi is None:
        roi=np.ones(gray.shape,dtype=bool);roi[int(gray.shape[0]*.91):]=False
    if roi.shape!=gray.shape:raise ValueError('选区尺寸与图片不一致。')
    interior=ndi.binary_erosion(roi,iterations=2)
    values=ndi.laplace(gray,mode='reflect')[interior]
    if values.size<64:raise ValueError('选区太小，无法估算阈值。')
    median=np.median(values);sigma=float(np.median(np.abs(values-median))/(0.67448975*np.sqrt(20.)))
    return round(float(np.clip(.65*sigma,1.,15.)),1),sigma


def segment_inside_roi(a, b, interior):
    """All pixels sampled along an intercept must remain inside the eroded ROI."""
    length=float(np.hypot(b[0]-a[0],b[1]-a[1]))
    count=max(2,int(np.ceil(length*4))+1)
    xx=np.rint(np.linspace(a[0],b[0],count)).astype(int)
    yy=np.rint(np.linspace(a[1],b[1],count)).astype(int)
    h,w=interior.shape
    if np.any(xx<0) or np.any(xx>=w) or np.any(yy<0) or np.any(yy>=h):return False
    return bool(np.all(interior[yy,xx]))


def review_tiles(roi, divisions=4):
    """Cover every selected pixel with a human review tile, including detector misses."""
    ys,xs=np.nonzero(roi)
    if not len(xs):return []
    x0,x1=int(xs.min()),int(xs.max())+1
    y0,y1=int(ys.min()),int(ys.max())+1
    xe=np.unique(np.linspace(x0,x1,divisions+1,dtype=int))
    ye=np.unique(np.linspace(y0,y1,divisions+1,dtype=int))
    return [{'bbox':(int(a),int(c),int(b),int(d)),'kind':'tile','threshold':None,'coherence':None,'state':'待检查'}
            for c,d in zip(ye[:-1],ye[1:]) for a,b in zip(xe[:-1],xe[1:]) if roi[c:d,a:b].any()]


def find_uncertain_regions(image, roi, phase, min_width, max_width, min_length, sensitivity, measured_candidates, coherence_min=0.7):
    """Flag elongated segmented components whose thickness could not be measured.

    These are review regions, not classified lamellae and not independent measurements.
    The image outside the boxes is not certified free of lamellae.
    """
    if not np.isfinite(coherence_min) or not 0<=coherence_min<=1:raise ValueError('方向一致性须在 0 到 1 之间。')
    gray=np.asarray(image.convert('L'),dtype=np.float32)/255
    if roi.shape!=gray.shape:raise ValueError('选区尺寸与图片不一致。')
    smooth=ndi.gaussian_filter(gray,.8);background=ndi.gaussian_filter(smooth,max(6.,max_width*1.5))
    contrast=smooth-background
    binary=((contrast < -sensitivity/255) if phase=='dark' else (contrast > sensitivity/255)) & roi
    binary=morphology.remove_small_objects(binary,min_size=max(8,int(min_width*min_length*.3)))
    labels=measure.label(binary,connectivity=2)
    gx=ndi.gaussian_filter(gray,1,order=(0,1));gy=ndi.gaussian_filter(gray,1,order=(1,0));sigma=max(6.,max_width*.9)
    xx=ndi.gaussian_filter(gx*gx,sigma);yy=ndi.gaussian_filter(gy*gy,sigma);xy=ndi.gaussian_filter(gx*gy,sigma)
    coherence=np.hypot(xx-yy,2*xy)/(xx+yy+1e-12)
    measured={tuple(row['bbox']) for row in measured_candidates if row.get('bbox') is not None}
    regions=[]
    for prop in measure.regionprops(labels):
        y0,x0,y1,x1=prop.bbox;box=(int(x0),int(y0),int(x1),int(y1));cy,cx=np.rint(prop.centroid).astype(int)
        if box in measured or prop.axis_major_length < .8*min_length:continue
        if prop.area/max(prop.axis_major_length,1)>2*max_width:continue
        score=float(coherence[cy,cx])
        if score < coherence_min:continue
        regions.append({'bbox':box,'threshold':float(sensitivity),'coherence':score})
    return regions


def likely_same_lamella(first, second):
    """Heuristic for review warnings; never establishes identity or drops a candidate."""
    def trace(item):
        pts=np.array([[(a[0]+b[0])/2,(a[1]+b[1])/2] for a,b,_ in item['points']],dtype=float)
        delta=pts[-1]-pts[0];length=float(np.linalg.norm(delta))
        if length<5:return None
        return pts.mean(axis=0),delta/length,length
    left,right=trace(first),trace(second)
    if left is None or right is None:return False
    a,u,la=left;b,v,lb=right;alignment=abs(float(u@v))
    if alignment<np.cos(np.deg2rad(15)):return False
    difference=b-a
    # Evaluate both traces' axes so that the warning does not depend on which
    # candidate was selected first. It is intentionally a warning, not a merge.
    for axis in (u,v):
        normal=np.array([-axis[1],axis[0]])
        if abs(float(difference@normal))<=3.0 and abs(float(difference@axis)) <= (la+lb)/2-.3*min(la,lb):return True
    return False


def detect(image, roi=None, phase='dark', min_width=2.5, max_width=20., min_length=30., elongation=3., sensitivity=0., progress=None, coherence_min=0., boundary_method="mask", parallel_only=False, neighbor_width_factor=6.):
    """Return candidate segments and three normal intercepts; coordinates are image pixels."""
    if boundary_method not in ("mask","half"): raise ValueError("未知的边界方法")
    if not np.isfinite(neighbor_width_factor) or not 2<=neighbor_width_factor<=20: raise ValueError("邻距倍数须在 2 到 20 之间")
    gray=np.asarray(image.convert('L'),dtype=np.float32)/255
    h,w=gray.shape
    if roi is None:
        roi=np.ones((h,w),bool); roi[int(h*.91):]=False
    # Local contrast rather than global intensity accommodates illumination variations.
    smooth=ndi.gaussian_filter(gray, 0.8)
    background=ndi.gaussian_filter(smooth, max(6.,max_width*1.5))
    residual=smooth-background
    binary=(residual < -sensitivity/255) if phase=='dark' else (residual > sensitivity/255)
    binary &= roi
    binary=morphology.remove_small_objects(binary, min_size=max(8,int(min_width*min_length*.3)))
    labels=measure.label(binary,connectivity=2)
    output=[]
    if coherence_min > 0:
        gx=ndi.gaussian_filter(gray,1,order=(0,1));gy=ndi.gaussian_filter(gray,1,order=(1,0))
        sigma=max(6.,max_width*.9)
        xx=ndi.gaussian_filter(gx*gx,sigma);yy=ndi.gaussian_filter(gy*gy,sigma);xy=ndi.gaussian_filter(gx*gy,sigma)
        coherence=np.hypot(xx-yy,2*xy)/(xx+yy+1e-12)
    interior=ndi.binary_erosion(roi, iterations=1)
    for prop in measure.regionprops(labels):
        y0,x0,y1,x1=prop.bbox
        if prop.area < min_length*min_width*.35 or prop.major_axis_length < min_length: continue
        if prop.major_axis_length/max(prop.minor_axis_length,1)<elongation: continue
        mask=prop.image
        # A lamella may extend past the ROI. Only individual intercepts must stay clear of its boundary.
        skel=morphology.skeletonize(mask)
        coords=np.argwhere(skel)
        if len(coords)<min_length*.8: continue
        # Suppress short skeleton spurs before checking for real long branches.
        pruned=skel.copy()
        for _ in range(6):
            degree=ndi.convolve(pruned.astype(np.uint8),np.ones((3,3),np.uint8),mode='constant')-pruned
            pruned &= ~(pruned & (degree<=1))
        neighbors=ndi.convolve(pruned.astype(np.uint8),np.ones((3,3),np.uint8),mode='constant')-pruned
        if np.count_nonzero((neighbors>=3)&pruned)>max(2,len(coords)*.025): continue
        # Sort along principal axis, reject disconnected and kinked skeletons.
        xy=coords[:,::-1].astype(float)
        center=xy.mean(axis=0); _,_,vh=np.linalg.svd(xy-center,full_matrices=False); axis=vh[0]
        proj=(xy-center)@axis; order=np.argsort(proj); xy=xy[order]; proj=proj[order]
        if proj[-1]-proj[0]<min_length: continue
        if len(coords)/(proj[-1]-proj[0]+1)>1.45: continue
        points=[]
        edge_methods=[]
        for frac in (.2,.5,.8):
            j=int(np.argmin(np.abs(proj-(proj[0]+frac*(proj[-1]-proj[0])))))
            x,y=xy[j]; ix,iy=int(x),int(y)
            local=xy[np.abs(proj-proj[j])<max(5,min_length*.16)]
            if len(local)<5: break
            _,_,v=np.linalg.svd(local-local.mean(axis=0),full_matrices=False); tangent=v[0]; normal=np.array([-tangent[1],tangent[0]])
            if not mask[iy,ix]: break
            # Interpolate the 0.5 contour of the binary mask along the local normal.
            ends=[]
            for direction in (-1,1):
                steps=np.arange(.25,max_width*1.5+.25,.25)
                coords=np.vstack((y+normal[1]*steps*direction,x+normal[0]*steps*direction))
                samples=ndi.map_coordinates(mask.astype(np.float32),coords,order=1,mode='constant',cval=0)
                below=np.flatnonzero(samples<.5)
                if not len(below):
                    ends.append(max_width*1.5)
                else:
                    idx=int(below[0]);prev=1. if idx==0 else float(samples[idx-1]);current=float(samples[idx]);previous=0. if idx==0 else float(steps[idx-1])
                    ends.append(previous+.25*(prev-.5)/max(prev-current,1e-12))
            methods=["mask","mask"]
            if boundary_method == "half":
                # Estimate each boundary at the half-contrast crossing of the local intensity profile.
                refined=[]
                for side,(direction,end) in enumerate(zip((-1,1),ends)):
                    steps=np.arange(0.,min(max_width*1.5,end+max(3.,end*.75))+.125,.25)
                    px=x+x0+normal[0]*steps*direction;py=y+y0+normal[1]*steps*direction
                    profile=ndi.map_coordinates(smooth,np.vstack((py,px)),order=1,mode='nearest')
                    inside=float(profile[0]);outer=profile[steps>=min(end,max(1.,end*.75))]
                    if not len(outer):refined.append(end);continue
                    outside=float(np.max(outer) if phase=='dark' else np.min(outer))
                    if abs(outside-inside)<2/255:refined.append(end);continue
                    midpoint=(inside+outside)/2
                    crossed=np.flatnonzero(profile>=midpoint if phase=='dark' else profile<=midpoint)
                    if len(crossed)==0 or crossed[0]==0:refined.append(end);continue
                    k=int(crossed[0]);a0=float(profile[k-1]);a1=float(profile[k]);frac=(midpoint-a0)/(a1-a0) if abs(a1-a0)>1e-12 else 0
                    crossing=float(steps[k-1]+.25*frac)
                    if abs(crossing-end)<=3:
                        refined.append(crossing);methods[side]="half"
                    else:refined.append(end)
                ends=refined
            width=sum(ends)
            if not min_width<=width<=max_width: break
            a=(x+x0-normal[0]*ends[0],y+y0-normal[1]*ends[0]);b=(x+x0+normal[0]*ends[1],y+y0+normal[1]*ends[1])
            if not segment_inside_roi(a,b,interior): break
            points.append((a,b,width));edge_methods.append(methods)
        if len(points)!=3: continue
        direction_score=None
        if coherence_min > 0:
            samples=[]
            for a,b,_ in points:
                cx=int(round((a[0]+b[0])/2));cy=int(round((a[1]+b[1])/2))
                samples.append(float(coherence[cy,cx]))
            direction_score=float(np.mean(samples))
            if direction_score < coherence_min: continue
        widths=[p[2] for p in points]
        if max(widths)>2.3*min(widths): continue
        local_contrasts=[]
        for a,b,_ in points:
            a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);direction=(b-a)/np.linalg.norm(b-a);midpoint_pos=(a+b)/2
            positions=[midpoint_pos,a-2*direction,b+2*direction]
            values=ndi.map_coordinates(smooth,np.array([[q[1] for q in positions],[q[0] for q in positions]]),order=1,mode='nearest')*255
            local_contrasts.append(float(min(values[1]-values[0],values[2]-values[0])) if phase=='dark' else float(min(values[0]-values[1],values[0]-values[2])))
        contrast_score=float(np.median(local_contrasts))
        # Objects may cross the ROI, provided all three intercepts are complete.
        # Record the segmented component's outline for visual review, not as a verified phase boundary.
        contours=measure.find_contours(np.pad(mask,1),.5)
        contour=[]
        if contours:
            outline=measure.approximate_polygon(max(contours,key=len),tolerance=.65)
            contour=[(float(np.clip(q[1]-1+x0,0,w-1)),float(np.clip(q[0]-1+y0,0,h-1))) for q in outline]
        output.append({'points':points,'threshold':float(sensitivity),'edge_methods':edge_methods,'contrast_score':contrast_score,'length':float(proj[-1]-proj[0]),'bbox':(x0,y0,x1,y1),'coherence':direction_score, 'axis':axis, 'center':np.array([center[0]+x0,center[1]+y0]), 'contour':contour})
    # A parallel neighbor is evidence of a local lamellar array, not proof of phase identity.
    if output:
        centers=np.asarray([item['center'] for item in output]);axes=np.asarray([item['axis'] for item in output]);lengths=np.asarray([item['length'] for item in output]);widths=np.asarray([np.mean([p[2] for p in item['points']]) for item in output])
        for i,item in enumerate(output):
            axis=axes[i];normal=np.array([-axis[1],axis[0]]);delta=centers-centers[i]
            normal_dist=np.abs(delta@normal);long_dist=np.abs(delta@axis)
            aligned=np.abs(axes@axis)>=np.cos(np.deg2rad(20))
            overlap=long_dist<=(lengths[i]+lengths)/2-.3*np.minimum(lengths[i],lengths)
            nearby=(normal_dist>=2)&(normal_dist<=max(30,max_width*3))
            neighbors=aligned & overlap & nearby;neighbors[i]=False
            item['neighbor_count']=int(np.count_nonzero(neighbors))
            # Distant aligned edges of coarse features may pass the broad neighbor test.
            max_near=np.minimum(max(30,max_width*3),np.maximum(12,neighbor_width_factor*np.maximum(widths[i],widths)))
            compact=neighbors & (normal_dist<=max_near)
            item['compact_neighbor_count']=int(np.count_nonzero(compact))
        if parallel_only:output=[item for item in output if item['compact_neighbor_count']>0]
    return output
