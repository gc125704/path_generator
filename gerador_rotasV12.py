import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import math
import numpy as np
from osgeo import gdal
import xml.etree.ElementTree as ET
import heapq
import itertools
import time

# ==============================
# Configuração de cores de obstáculos
# ==============================
OBSTACLES = [(34, 177, 76), (0, 255, 0), (0, 0, 255)]  # Verde e Azul

# ==============================
# Classe Ponto
# ==============================
class Ponto:
    def __init__(self, nome, lat=None, lon=None):
        self.nome = nome
        self.lat = lat
        self.lon = lon
        self.x = 0
        self.y = 0
        self.tipo = "normal"  # normal, inicio, desvio

# ==============================
# Função para checar obstáculo
# ==============================
def is_obstacle_color(pixel):
    r, g, b = pixel[:3]
    for o in OBSTACLES:
        if all(abs(c - oc) <= 15 for c, oc in zip((r, g, b), o)):
            return True
    return False

# ==============================
# A* para encontrar caminho entre dois pontos
# ==============================
def astar_between_points(pix, img_width, img_height, x1, y1, x2, y2, step_size=3):
    """
    Encontra caminho entre dois pontos usando A*, desviando de obstáculos.
    """
    start = (int(x1), int(y1))
    goal = (int(x2), int(y2))
    
    if is_obstacle_color(pix[start[0], start[1]]) or is_obstacle_color(pix[goal[0], goal[1]]):
        return None, float('inf')
    
    def heuristic(pos):
        return math.hypot(pos[0] - goal[0], pos[1] - goal[1])
    
    def get_neighbors(pos):
        neighbors = []
        for dx in [-step_size, 0, step_size]:
            for dy in [-step_size, 0, step_size]:
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = pos[0] + dx, pos[1] + dy
                
                if 0 <= nx < img_width and 0 <= ny < img_height:
                    if not is_obstacle_color(pix[nx, ny]):
                        cost = math.hypot(dx, dy)
                        neighbors.append(((nx, ny), cost))
        
        return neighbors
    
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start), 0, start))
    
    came_from = {}
    g_score = {start: 0}
    
    visited = set()
    max_iterations = 50000
    iterations = 0
    
    while open_set and iterations < max_iterations:
        iterations += 1
        _, current_g, current = heapq.heappop(open_set)
        
        if current in visited:
            continue
        visited.add(current)
        
        if math.hypot(current[0] - goal[0], current[1] - goal[1]) <= step_size * 1.5:
            path = []
            node = current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            path.append(goal)
            
            total_dist = sum(math.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]) 
                           for i in range(len(path)-1))
            return path, total_dist
        
        for neighbor, move_cost in get_neighbors(current):
            tentative_g = current_g + move_cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))
    
    return None, float('inf')

# ==============================
# Checar se linha cruza obstáculo
# ==============================
def has_obstacle_between(pix, x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    while True:
        if is_obstacle_color(pix[x1, y1]):
            return True
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return False

# ==============================
# NOVA FUNÇÃO: Simplificar caminho A* e gerar pontos de desvio
# ==============================
def simplify_path_and_create_detour_points(path, transform, min_distance_pixels=100, mode="complete", original_image=None):
    """
    Simplifica um caminho A* e cria pontos de desvio APENAS onde há obstáculos próximos.
    
    Args:
        path: Lista de tuplas (x, y) representando o caminho
        transform: Transformação geoespacial GDAL
        min_distance_pixels: Distância mínima em pixels entre pontos de desvio (padrão: 100)
        mode: "complete" (padrão) ou "simple" (apenas início/fim)
        original_image: Imagem PIL para verificar obstáculos
    
    Returns:
        Lista de objetos Ponto representando os pontos de desvio
    """
    if not path or len(path) <= 2:
        return []
    
    def pixel_to_geo(x, y):
        lon = transform[0] + x * transform[1] + y * transform[2]
        lat = transform[3] + x * transform[4] + y * transform[5]
        return lon, lat
    
    def is_near_obstacle(x, y, pix, radius=20):
        """
        Verifica se há obstáculos próximos ao ponto.
        Retorna True se encontrar obstáculo em um raio de pixels.
        """
        if pix is None:
            return False
        
        img_width = original_image.width
        img_height = original_image.height
        
        # Verifica em um círculo ao redor do ponto
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                # Verifica se está dentro do círculo
                if dx*dx + dy*dy > radius*radius:
                    continue
                
                check_x = int(x + dx)
                check_y = int(y + dy)
                
                # Verifica limites da imagem
                if 0 <= check_x < img_width and 0 <= check_y < img_height:
                    if is_obstacle_color(pix[check_x, check_y]):
                        return True
        
        return False
    
    def calculate_angle_change(p1, p2, p3):
        """Calcula a mudança de ângulo entre três pontos consecutivos"""
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        len1 = math.hypot(v1[0], v1[1])
        len2 = math.hypot(v2[0], v2[1])
        if len1 == 0 or len2 == 0:
            return 0
        
        dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)
        dot = max(-1, min(1, dot))
        
        return math.degrees(math.acos(dot))
    
    detour_points = []
    detour_counter = [1]
    
    # Carrega pixels se imagem disponível
    pix = original_image.load() if original_image else None
    
    # MODO SIMPLES: Apenas início e fim (se próximos de obstáculos)
    if mode == "simple":
        # Primeiro ponto
        x, y = path[0]
        if is_near_obstacle(x, y, pix, radius=30):
            lon, lat = pixel_to_geo(x, y)
            p = Ponto(f"D{detour_counter[0]}", lat, lon)
            p.x, p.y = x, y
            p.tipo = "desvio"
            detour_points.append(p)
            detour_counter[0] += 1
        
        # Último ponto
        x, y = path[-1]
        if is_near_obstacle(x, y, pix, radius=30):
            lon, lat = pixel_to_geo(x, y)
            p = Ponto(f"D{detour_counter[0]}", lat, lon)
            p.x, p.y = x, y
            p.tipo = "desvio"
            detour_points.append(p)
        
        return detour_points
    
    # MODO COMPLETO: Gera pontos APENAS onde há obstáculos próximos
    last_added = path[0]
    
    # Pontos intermediários (ignora primeiro e último)
    for i in range(1, len(path) - 1):
        current = path[i]
        
        dist_from_last = math.hypot(current[0] - last_added[0], current[1] - last_added[1])
        
        # Verifica se está perto de obstáculo
        near_obstacle = is_near_obstacle(current[0], current[1], pix, radius=25)
        
        # Verifica mudança de ângulo
        angle_change = 0
        if i > 0 and i < len(path) - 1:
            angle_change = calculate_angle_change(path[i-1], current, path[i+1])
        
        # NOVO CRITÉRIO: Só adiciona se:
        # 1. Está PERTO de obstáculo E
        # 2. (Distância mínima atingida OU curva acentuada)
        if near_obstacle and (dist_from_last >= min_distance_pixels or angle_change > 45):
            x, y = current
            lon, lat = pixel_to_geo(x, y)
            p = Ponto(f"D{detour_counter[0]}", lat, lon)
            p.x, p.y = x, y
            p.tipo = "desvio"
            detour_points.append(p)
            detour_counter[0] += 1
            last_added = current
    
    return detour_points

# ==============================
# Aplicativo principal
# ==============================
class PathPlannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Planejamento de Caminho - Com Pontos de Desvio")
        self.root.geometry("1200x700")

        self.points = {"inicio": None, "normais": [], "desvios": []}
        self.mode = "normal"
        self.path = []
        self.detailed_paths = {}
        self.selected_algorithm = tk.StringVar(value="A* TSP (Ótimo/Rápido)")

        # Frames
        self.frame_controls = ttk.Frame(root, padding=10)
        self.frame_controls.pack(side=tk.LEFT, fill=tk.Y)
        self.frame_canvas = ttk.Frame(root)
        self.frame_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Widgets de controle
        ttk.Button(self.frame_controls, text="Carregar Imagem", command=self.load_image).pack(fill=tk.X, pady=5)
        ttk.Button(self.frame_controls, text="Definir Início (Azul)", command=lambda: self.set_mode("inicio")).pack(fill=tk.X, pady=5)
        ttk.Button(self.frame_controls, text="Adicionar Ponto (Vermelho)", command=lambda: self.set_mode("normal")).pack(fill=tk.X, pady=5)
        ttk.Button(self.frame_controls, text="Remover Ponto Vermelho", command=lambda: self.set_mode("remover")).pack(fill=tk.X, pady=5)

        ttk.Label(self.frame_controls, text="Algoritmo de Caminho:").pack(pady=(20,0))
        algoritmo_menu = ttk.Combobox(self.frame_controls, textvariable=self.selected_algorithm,
                                      values=["Guloso (Rápido)", "A* TSP (Ótimo/Rápido)", "Dijkstra TSP (Ótimo/Lento)"], state="readonly")
        algoritmo_menu.pack(fill=tk.X, pady=5)

        ttk.Button(self.frame_controls, text="Gerar Caminho", command=self.generate_path).pack(fill=tk.X, pady=5)
        ttk.Button(self.frame_controls, text="Visualizar Conexões", command=self.show_connections).pack(fill=tk.X, pady=5)
        ttk.Button(self.frame_controls, text="Exportar XML", command=self.export_to_xml).pack(fill=tk.X, pady=5)
        ttk.Button(self.frame_controls, text="Exportar Pontos (sem rota)", command=self.export_points_only).pack(fill=tk.X, pady=5)
        ttk.Button(self.frame_controls, text="Importar XML", command=self.import_from_xml).pack(fill=tk.X, pady=5)

        # Info do caminho
        self.label_distancia = ttk.Label(self.frame_controls, text="Distância: -")
        self.label_distancia.pack(pady=(20,0))
        
        self.label_final = ttk.Label(self.frame_controls, text="Final escolhido: -", foreground="green")
        self.label_final.pack(pady=5)
        
        self.label_strategy = ttk.Label(self.frame_controls, text="Estratégia: -", foreground="blue")
        self.label_strategy.pack(pady=5)
        
        self.label_desvios = ttk.Label(self.frame_controls, text="Pontos de desvio: 0", foreground="orange")
        self.label_desvios.pack(pady=5)

        self.label_tempo = ttk.Label(self.frame_controls, text="Tempo de execução: -", foreground="purple")
        self.label_tempo.pack(pady=5)

        # Legenda
        ttk.Label(self.frame_controls, text="Legenda:").pack(pady=(20,0))
        ttk.Label(self.frame_controls, text="● Azul = Início").pack(anchor="w")
        ttk.Label(self.frame_controls, text="● Verde = Final (automático)").pack(anchor="w")
        ttk.Label(self.frame_controls, text="● Vermelho = Pontos").pack(anchor="w")
        ttk.Label(self.frame_controls, text="■ Laranja = Desvios").pack(anchor="w")
        ttk.Label(self.frame_controls, text="— Preto = Caminho").pack(anchor="w")

        # Canvas
        self.canvas = tk.Canvas(self.frame_canvas, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.frame_canvas.bind("<Configure>", self.resize_image)
        
        self.label_coords = ttk.Label(self.frame_canvas, text="Coordenadas: Lat: -, Lon: -", 
                                     relief=tk.SUNKEN, anchor=tk.W)
        self.label_coords.pack(side=tk.BOTTOM, fill=tk.X)

        # Imagens
        self.original_image = None
        self.display_image = None
        self.tk_img = None
        self.img_on_canvas = None
        self.transform = None
        self.projection = None

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Imagens TIFF","*.tiff"), ("PNG","*.png"), ("Todos", "*.*")])
        if not file_path:
            return
        ds = gdal.Open(file_path)
        if ds is None:
            messagebox.showerror("Erro","Erro ao abrir imagem com GDAL")
            return
        self.transform = ds.GetGeoTransform()
        self.projection = ds.GetProjection()  # WKT da projeção para detectar unidade
        print(f"[INFO] GeoTransform: {self.transform}")
        print(f"[INFO] Pixel size X: {abs(self.transform[1]):.8f}, Y: {abs(self.transform[5]):.8f}")
        img_array = ds.ReadAsArray()
        if img_array.ndim == 3:
            img_array = np.transpose(img_array,(1,2,0))
        elif img_array.ndim == 2:
            img_array = np.stack([img_array]*3, axis=-1)
        if img_array.shape[2] >=3:
            img_array = img_array[...,:3]
        self.original_image = Image.fromarray(img_array.astype(np.uint8))
        self.resize_image()

    def resize_image(self,event=None):
        if not self.original_image:
            return
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w<10 or canvas_h<10:
            return
        self.display_image = self.original_image.resize((canvas_w,canvas_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(self.display_image)
        if self.img_on_canvas:
            self.canvas.delete(self.img_on_canvas)
        self.img_on_canvas = self.canvas.create_image(0,0,anchor=tk.NW,image=self.tk_img)
        self.redraw_points()
        self.redraw_path()

    def set_mode(self, mode):
        self.mode = mode

    def pixel_to_geo(self, x, y):
        if not self.transform:
            return 0, 0
        lon = self.transform[0] + x * self.transform[1] + y * self.transform[2]
        lat = self.transform[3] + x * self.transform[4] + y * self.transform[5]
        return lon, lat

    def geo_to_pixel(self, lon, lat):
        if not self.transform:
            raise ValueError("Transformação geoespacial não definida.")
        try:
            inv_gt = gdal.InvGeoTransform(self.transform)
        except Exception:
            success, inv_gt = gdal.InvGeoTransform(self.transform)
            if not success:
                raise ValueError("Falha ao inverter a transformação geoespacial.")
        x = inv_gt[0] + inv_gt[1] * lon + inv_gt[2] * lat
        y = inv_gt[3] + inv_gt[4] * lon + inv_gt[5] * lat
        return int(x), int(y)

    def pixels_to_meters(self, pixels):
        """
        Converte distância em pixels para metros usando o transform GDAL.

        - Projeção métrica (UTM, SIRGAS, Mercator, etc.): transform[1] já é m/pixel.
        - Projeção geográfica (graus, ex. EPSG:4326): cada eixo é convertido
          separadamente para metros antes de calcular o tamanho médio do pixel.
        """
        if not self.transform:
            return None

        pixel_size_x = abs(self.transform[1])   # graus/pixel ou metros/pixel em X
        pixel_size_y = abs(self.transform[5])   # graus/pixel ou metros/pixel em Y

        # Detecta se a projeção é geográfica (graus) ou projetada (metros)
        is_geographic = True  # conservador: assume graus
        if hasattr(self, 'projection') and self.projection:
            try:
                from osgeo import osr
                srs = osr.SpatialReference()
                srs.ImportFromWkt(self.projection)
                if srs.IsProjected():
                    is_geographic = False  # UTM, Mercator, etc. → já em metros
            except Exception:
                pass

        if not is_geographic:
            # Unidade já é metros — tamanho médio do pixel × número de pixels
            m_per_pixel = (pixel_size_x + pixel_size_y) / 2.0
            return pixels * m_per_pixel
        else:
            # Unidade é graus → converte cada eixo para metros separadamente
            if self.original_image:
                cx = self.original_image.width / 2
                cy = self.original_image.height / 2
                _, lat_center = self.pixel_to_geo(cx, cy)
            else:
                lat_center = 0.0

            lat_rad = math.radians(lat_center)
            # 1 grau de latitude  ≈ 111 320 m (constante)
            # 1 grau de longitude ≈ 111 320 × cos(lat) m
            m_per_deg_lat = 111320.0
            m_per_deg_lon = 111320.0 * math.cos(lat_rad)

            # Tamanho do pixel em metros por eixo
            m_per_pixel_x = pixel_size_x * m_per_deg_lon
            m_per_pixel_y = pixel_size_y * m_per_deg_lat
            m_per_pixel = (m_per_pixel_x + m_per_pixel_y) / 2.0

            return pixels * m_per_pixel

    def format_distance(self, pixels):
        """Retorna string formatada da distância em metros ou km."""
        metros = self.pixels_to_meters(pixels)
        if metros is None:
            return f"{pixels:.1f} px"
        if metros >= 1000:
            return f"{metros/1000:.3f} km"
        return f"{metros:.1f} m"

    def on_click(self,event):
        if not self.original_image or not self.display_image:
            return
        scale_x = self.original_image.width / self.display_image.width
        scale_y = self.original_image.height / self.display_image.height
        x_img = int(event.x*scale_x)
        y_img = int(event.y*scale_y)
        lon,lat = self.pixel_to_geo(x_img,y_img)
        nome = f"P{len(self.points['normais'])+1}"

        if self.mode=="inicio":
            p = Ponto("inicio",lat,lon)
            p.x,p.y,p.tipo = x_img,y_img,"inicio"
            self.points["inicio"]=p
        elif self.mode=="normal":
            p = Ponto(nome,lat,lon)
            p.x,p.y,p.tipo = x_img,y_img,"normal"
            self.points["normais"].append(p)
        elif self.mode=="remover":
            for p in self.points["normais"]:
                if abs(p.x-x_img)<5 and abs(p.y-y_img)<5:
                    self.points["normais"].remove(p)
                    break
        self.redraw_points()

    def on_mouse_move(self, event):
        if not self.original_image or not self.display_image or not self.transform:
            return
        
        scale_x = self.original_image.width / self.display_image.width
        scale_y = self.original_image.height / self.display_image.height
        x_img = int(event.x * scale_x)
        y_img = int(event.y * scale_y)
        
        if 0 <= x_img < self.original_image.width and 0 <= y_img < self.original_image.height:
            lon, lat = self.pixel_to_geo(x_img, y_img)
            self.label_coords.config(text=f"Coordenadas: Lat: {lat:.6f}, Lon: {lon:.6f}")
        else:
            self.label_coords.config(text="Coordenadas: Lat: -, Lon: -")

    def redraw_points(self):
        self.canvas.delete("point")
        if not self.original_image or not self.display_image:
            return
        scale_x = self.display_image.width / self.original_image.width
        scale_y = self.display_image.height / self.original_image.height

        def to_canvas(p):
            return int(p.x*scale_x), int(p.y*scale_y)

        # IMPORTANTE: Desenha na ordem correta (de trás para frente)
        
        # 1. Primeiro desenha pontos de desvio (ATRÁS)
        print(f"[DEBUG redraw_points] Desenhando {len(self.points['desvios'])} pontos de desvio")
        for idx, p in enumerate(self.points["desvios"]):
            x, y = to_canvas(p)
            print(f"  Desvio {idx+1}: {p.nome} na posição canvas ({x}, {y}) - original ({p.x}, {p.y})")
            self.canvas.create_rectangle(x-4, y-4, x+4, y+4, 
                                        fill="orange", 
                                        outline="darkorange", 
                                        width=2, 
                                        tags="point")
        
        # 2. Depois desenha ponto de início (MEIO)
        if self.points["inicio"]:
            x,y = to_canvas(self.points["inicio"])
            self.canvas.create_oval(x-5,y-5,x+5,y+5,fill="blue",tags="point")
        
        final_point = None
        if self.path and len(self.path) > 0:
            final_point = self.path[-1]
        
        # 3. Por último desenha pontos normais (FRENTE) - sobrepõem os desvios
        for p in self.points["normais"]:
            x,y = to_canvas(p)
            if final_point and p.nome == final_point.nome:
                self.canvas.create_oval(x-5,y-5,x+5,y+5,fill="green",tags="point")
            else:
                self.canvas.create_oval(x-3,y-3,x+3,y+3,fill="red",tags="point")

    def redraw_path(self):
        self.canvas.delete("path")
        if not self.path or not self.original_image or not self.display_image:
            return
        
        scale_x = self.display_image.width / self.original_image.width
        scale_y = self.display_image.height / self.original_image.height
        
        for i in range(len(self.path)-1):
            p1 = self.path[i]
            p2 = self.path[i+1]
            
            key = (p1.nome, p2.nome)
            if key in self.detailed_paths and self.detailed_paths[key]:
                detailed_path = self.detailed_paths[key]
                
                for j in range(len(detailed_path)-1):
                    x1 = int(detailed_path[j][0] * scale_x)
                    y1 = int(detailed_path[j][1] * scale_y)
                    x2 = int(detailed_path[j+1][0] * scale_x)
                    y2 = int(detailed_path[j+1][1] * scale_y)
                    self.canvas.create_line(x1, y1, x2, y2, fill="black", width=2, tags="path")
            else:
                x1 = int(p1.x * scale_x)
                y1 = int(p1.y * scale_y)
                x2 = int(p2.x * scale_x)
                y2 = int(p2.y * scale_y)
                self.canvas.create_line(x1, y1, x2, y2, fill="black", width=2, tags="path")

    def show_connections(self):
        if not self.points["inicio"]:
            messagebox.showwarning("Aviso","Defina início primeiro!")
            return
        
        if len(self.points["normais"]) < 2:
            messagebox.showwarning("Aviso","Adicione pelo menos 2 pontos!")
            return
        
        self.canvas.delete("connections")
        
        distances, pontos, _ = self.build_complete_graph("antiga")
        
        if not self.original_image or not self.display_image:
            return
        
        scale_x = self.display_image.width / self.original_image.width
        scale_y = self.display_image.height / self.original_image.height
        
        count = 0
        for i, a in enumerate(pontos):
            for b in pontos[i+1:]:
                if distances[a][b] != float('inf'):
                    x1 = int(a.x * scale_x)
                    y1 = int(a.y * scale_y)
                    x2 = int(b.x * scale_x)
                    y2 = int(b.y * scale_y)
                    self.canvas.create_line(x1, y1, x2, y2, 
                                          fill="lightgray", width=1, 
                                          tags="connections")
                    count += 1
        
        self.canvas.tag_lower("connections")
        
        messagebox.showinfo("Conexões", 
                          f"Visualizando {count} conexões possíveis\n"
                          f"Veja o console para detalhes.")

    def build_complete_graph(self, strategy="antiga"):
        """
        Constrói grafo com diferentes estratégias.
        IMPORTANTE: NÃO gera pontos de desvio aqui (apenas na fase final).
        """
        pix = self.original_image.load()
        pontos = [self.points["inicio"]] + self.points["normais"]
        img_width = self.original_image.width
        img_height = self.original_image.height
        
        distances = {}
        paths_dict = {}
        total_connections = 0
        blocked_connections = 0
        direct_lines = 0
        astar_paths = 0
        
        print(f"\n{'='*60}")
        print(f"CONSTRUINDO GRAFO - Estratégia: {strategy.upper()}")
        print(f"{'='*60}")
        print(f"Total de pontos: {len(pontos)}")
        print(f"Pares a calcular: {len(pontos)*(len(pontos)-1)//2}")
        
        for i, a in enumerate(pontos):
            distances[a] = {}
            for j, b in enumerate(pontos):
                if i == j:
                    distances[a][b] = 0
                    continue
                
                if i > j:
                    distances[a][b] = distances[b][a]
                    if (b.nome, a.nome) in paths_dict:
                        paths_dict[(a.nome, b.nome)] = list(reversed(paths_dict[(b.nome, a.nome)]))
                    continue
                
                # ESTRATÉGIA ANTIGA: Apenas linha reta
                if strategy == "antiga":
                    if not has_obstacle_between(pix, int(a.x), int(a.y), int(b.x), int(b.y)):
                        dist = math.hypot(a.x - b.x, a.y - b.y)
                        distances[a][b] = dist
                        paths_dict[(a.nome, b.nome)] = [(int(a.x), int(a.y)), (int(b.x), int(b.y))]
                        total_connections += 1
                        direct_lines += 1
                    else:
                        distances[a][b] = float('inf')
                        blocked_connections += 1
                
                # ESTRATÉGIA HÍBRIDA
                elif strategy == "hibrido":
                    if not has_obstacle_between(pix, int(a.x), int(a.y), int(b.x), int(b.y)):
                        dist = math.hypot(a.x - b.x, a.y - b.y)
                        distances[a][b] = dist
                        paths_dict[(a.nome, b.nome)] = [(int(a.x), int(a.y)), (int(b.x), int(b.y))]
                        total_connections += 1
                        direct_lines += 1
                        print(f"  {a.nome} ↔ {b.nome}: linha reta ({dist:.1f})")
                    else:
                        print(f"  {a.nome} ↔ {b.nome}: bloqueado, tentando A*...")
                        path, dist = astar_between_points(pix, img_width, img_height, 
                                                         a.x, a.y, b.x, b.y, step_size=5)
                        
                        if path and dist != float('inf'):
                            distances[a][b] = dist
                            paths_dict[(a.nome, b.nome)] = path
                            total_connections += 1
                            astar_paths += 1
                            
                            print(f"    ✓ Contorno A* ({dist:.1f} px, {len(path)} pts)")
                        else:
                            distances[a][b] = float('inf')
                            blocked_connections += 1
                            print(f"    ✗ Sem caminho")
                
                # ESTRATÉGIA SEMPRE A*
                else:  # sempre_astar
                    print(f"  {a.nome} ↔ {b.nome}: calculando com A*...")
                    path, dist = astar_between_points(pix, img_width, img_height, 
                                                     a.x, a.y, b.x, b.y, step_size=5)
                    
                    if path and dist != float('inf'):
                        distances[a][b] = dist
                        paths_dict[(a.nome, b.nome)] = path
                        total_connections += 1
                        astar_paths += 1
                        
                        print(f"    ✓ Caminho A* ({dist:.1f} px, {len(path)} pts)")
                    else:
                        distances[a][b] = float('inf')
                        blocked_connections += 1
                        print(f"    ✗ Sem caminho")
        
        print(f"\n{'='*60}")
        print(f"ESTATÍSTICAS DO GRAFO")
        print(f"{'='*60}")
        if strategy != "antiga":
            print(f"Linhas retas: {direct_lines}")
            print(f"Caminhos A*: {astar_paths}")
        print(f"Total conectado: {total_connections}")
        print(f"Total bloqueado: {blocked_connections}")
        if total_connections + blocked_connections > 0:
            conectividade = total_connections/(total_connections+blocked_connections)*100
            print(f"Conectividade: {conectividade:.1f}%")
        
        return distances, pontos, paths_dict

    def generate_detour_points_for_final_path(self):
        """
        Gera pontos de desvio APENAS para os segmentos do caminho final
        que usaram A* (não linha reta).
        """
        if not self.path or len(self.path) < 2:
            print("⚠️ Nenhum caminho para gerar pontos de desvio")
            return
        
        self.points["desvios"] = []  # Limpa pontos anteriores
        
        print(f"\n{'='*60}")
        print(f"GERANDO PONTOS DE DESVIO PARA CAMINHO FINAL")
        print(f"{'='*60}")
        print(f"Caminho final tem {len(self.path)} pontos")
        print(f"Detailed_paths tem {len(self.detailed_paths)} chaves")
        print(f"Chaves disponíveis: {list(self.detailed_paths.keys())}")
        
        for i in range(len(self.path) - 1):
            p1 = self.path[i]
            p2 = self.path[i + 1]
            
            key = (p1.nome, p2.nome)
            print(f"\n  Processando segmento: {p1.nome} → {p2.nome}")
            print(f"    Procurando chave: {key}")
            
            # Verifica se existe caminho detalhado (A*) para este segmento
            if key in self.detailed_paths:
                path = self.detailed_paths[key]
                print(f"    Caminho encontrado com {len(path)} pontos")
                
                # Verifica se é realmente um desvio (mais de 2 pontos = não é linha reta)
                if len(path) > 2:
                    print(f"    É um desvio! Gerando pontos...")
                    
                    if self.transform:
                        detour_points = simplify_path_and_create_detour_points(
                            path, self.transform, 
                            min_distance_pixels=100,
                            mode="complete",
                            original_image=self.original_image  # Passa a imagem
                        )
                        
                        self.points["desvios"].extend(detour_points)
                        print(f"    ✓ {len(detour_points)} pontos de desvio gerados")
                        
                        # Debug: mostra os pontos gerados
                        for dp in detour_points:
                            print(f"      - {dp.nome}: ({dp.x}, {dp.y}) lat={dp.lat:.6f}, lon={dp.lon:.6f}")
                    else:
                        print(f"    ✗ Transform não disponível!")
                else:
                    print(f"    Linha reta (apenas {len(path)} pontos)")
            else:
                print(f"    ✗ Chave não encontrada em detailed_paths")
        
        print(f"\n{'='*60}")
        print(f"Total de pontos de desvio gerados: {len(self.points['desvios'])}")
        print(f"{'='*60}\n")

    def greedy_tsp_all_ends(self, distances, start, candidates):
        best_path = None
        best_distance = float('inf')
        best_end = None
        
        for end_candidate in candidates:
            intermediates = [p for p in candidates if p != end_candidate]
            path = self.greedy_tsp(distances, start, end_candidate, intermediates)
            
            if path:
                total_dist = sum(distances[path[i]][path[i+1]] for i in range(len(path)-1))
                
                if total_dist < best_distance:
                    best_distance = total_dist
                    best_path = path
                    best_end = end_candidate
        
        return best_path, best_distance, best_end

    def greedy_tsp(self, distances, start, end, intermediates):
        attempts = [0]
        
        def try_greedy(current, unvisited, path):
            attempts[0] += 1
            if attempts[0] > 10000:
                return None
            
            if not unvisited:
                if distances[current][end] != float('inf'):
                    return path + [end]
                return None
            
            accessible = [(p, distances[current][p]) for p in unvisited 
                         if distances[current][p] != float('inf')]
            
            if not accessible:
                return None
            
            accessible.sort(key=lambda x: x[1])
            
            for next_point, dist in accessible:
                new_unvisited = unvisited - {next_point}
                new_path = path + [next_point]
                result = try_greedy(next_point, new_unvisited, new_path)
                if result:
                    return result
            
            return None
        
        return try_greedy(start, set(intermediates), [start])

    def dijkstra_tsp_all_ends(self, distances, start, candidates):
        best_path = None
        best_distance = float('inf')
        best_end = None
        
        for end_candidate in candidates:
            intermediates = [p for p in candidates if p != end_candidate]
            path = self.dijkstra_tsp(distances, start, end_candidate, intermediates)
            
            if path:
                total_dist = sum(distances[path[i]][path[i+1]] for i in range(len(path)-1))
                
                if total_dist < best_distance:
                    best_distance = total_dist
                    best_path = path
                    best_end = end_candidate
        
        return best_path, best_distance, best_end

    def dijkstra_tsp(self, distances, start, end, intermediates):
        n_intermediates = len(intermediates)
        MAX_STATES = 50000
        
        counter = itertools.count()
        initial_state = (0, next(counter), start, frozenset([start]), [start])
        pq = [initial_state]
        
        best_cost = {}
        visited_full_states = set()
        
        while pq:
            if len(pq) > MAX_STATES:
                pq.sort()
                pq = pq[:MAX_STATES//2]
            
            cost, _, current, visited, path = heapq.heappop(pq)
            
            state_hash = (current, visited)
            if state_hash in visited_full_states:
                continue
            visited_full_states.add(state_hash)
            
            visited_intermediates = visited - {start, end}
            
            if len(visited_intermediates) == n_intermediates:
                if current == end:
                    return path
                if distances[current][end] != float('inf'):
                    return path + [end]
            
            for next_point in intermediates:
                if next_point not in visited and distances[current][next_point] != float('inf'):
                    new_cost = cost + distances[current][next_point]
                    new_visited = visited | frozenset([next_point])
                    new_path = path + [next_point]
                    state_key = (next_point, new_visited)
                    
                    if state_key not in best_cost or new_cost < best_cost[state_key]:
                        best_cost[state_key] = new_cost
                        heapq.heappush(pq, (new_cost, next(counter), next_point, new_visited, new_path))
            
            if len(best_cost) > MAX_STATES * 2:
                items = list(best_cost.items())
                items.sort(key=lambda x: (len(x[0][1]), -x[1]), reverse=True)
                best_cost = dict(items[:MAX_STATES])
        
        return None

    def astar_tsp_all_ends(self, distances, start, candidates):
        best_path = None
        best_distance = float('inf')
        best_end = None
        
        for end_candidate in candidates:
            intermediates = [p for p in candidates if p != end_candidate]
            path = self.astar_tsp(distances, start, end_candidate, intermediates)
            
            if path:
                total_dist = sum(distances[path[i]][path[i+1]] for i in range(len(path)-1))
                
                if total_dist < best_distance:
                    best_distance = total_dist
                    best_path = path
                    best_end = end_candidate
        
        return best_path, best_distance, best_end

    def astar_tsp(self, distances, start, end, intermediates):
        n_intermediates = len(intermediates)
        
        def mst_heuristic(current, unvisited, end_point):
            if not unvisited:
                return distances[current][end_point]
            
            min_to_unvisited = min((distances[current][p] for p in unvisited 
                                   if distances[current][p] != float('inf')), default=float('inf'))
            if min_to_unvisited == float('inf'):
                return float('inf')
            
            mst_cost = 0
            if len(unvisited) > 1:
                temp_unvisited = list(unvisited)
                for i in range(len(temp_unvisited) - 1):
                    min_dist = float('inf')
                    for j in range(i + 1, len(temp_unvisited)):
                        d = distances[temp_unvisited[i]][temp_unvisited[j]]
                        if d < min_dist:
                            min_dist = d
                    if min_dist != float('inf'):
                        mst_cost += min_dist
            
            min_to_end = min((distances[p][end_point] for p in unvisited 
                            if distances[p][end_point] != float('inf')), default=float('inf'))
            
            return min_to_unvisited + mst_cost + min_to_end
        
        counter = itertools.count()
        initial_unvisited = frozenset(intermediates)
        h_initial = mst_heuristic(start, initial_unvisited, end)
        
        pq = [(h_initial, 0, next(counter), start, initial_unvisited, [start])]
        visited_states = set()
        
        while pq:
            f, g, _, current, unvisited, path = heapq.heappop(pq)
            
            state = (current, unvisited)
            if state in visited_states:
                continue
            visited_states.add(state)
            
            if not unvisited:
                if current == end:
                    return path
                
                if distances[current][end] != float('inf'):
                    return path + [end]
            
            for next_point in unvisited:
                if distances[current][next_point] != float('inf'):
                    new_g = g + distances[current][next_point]
                    new_unvisited = unvisited - frozenset([next_point])
                    new_h = mst_heuristic(next_point, new_unvisited, end)
                    
                    if new_h == float('inf'):
                        continue
                    
                    new_f = new_g + new_h
                    new_path = path + [next_point]
                    
                    heapq.heappush(pq, (new_f, new_g, next(counter), next_point, new_unvisited, new_path))
        
        return None

    def generate_path(self):
        if not self.points["inicio"]:
            messagebox.showwarning("Aviso","Defina o ponto de início!")
            return
        
        if len(self.points["normais"]) < 2:
            messagebox.showwarning("Aviso","Adicione pelo menos 2 pontos!")
            return
        
        start = self.points["inicio"]
        candidates = self.points["normais"]
        algo = self.selected_algorithm.get()
        
        tempo_inicio = time.time()
        
        print(f"\n{'#'*60}")
        print(f"# INICIANDO GERAÇÃO DE CAMINHO")
        print(f"# Algoritmo TSP: {algo}")
        print(f"# Pontos: {len(candidates)+1} (início + {len(candidates)} normais)")
        print(f"{'#'*60}\n")
        
        # NÍVEL 1: ABORDAGEM ANTIGA
        self.label_distancia.config(text="Tentando abordagem rápida...")
        self.label_strategy.config(text="Estratégia: Linha reta")
        self.root.update()
        
        print("┌" + "─"*58 + "┐")
        print("│ NÍVEL 1: ABORDAGEM ANTIGA (Apenas linhas retas)         │")
        print("└" + "─"*58 + "┘")
        
        distances_old, _, paths_old = self.build_complete_graph("antiga")
        
        path_old = None
        distance_old = float('inf')
        chosen_end_old = None
        
        if "Guloso" in algo:
            path_old, distance_old, chosen_end_old = self.greedy_tsp_all_ends(distances_old, start, candidates)
        elif "Dijkstra" in algo:
            if len(candidates) > 12:
                resposta = messagebox.askyesno(
                    "Aviso de Performance",
                    f"Você tem {len(candidates)} pontos.\n\n"
                    "Dijkstra TSP pode levar muito tempo.\n"
                    "Recomendamos 'A* TSP'.\n\n"
                    "Continuar com Dijkstra?"
                )
                if not resposta:
                    self.label_distancia.config(text="Distância: -")
                    self.label_strategy.config(text="Estratégia: -")
                    return
            path_old, distance_old, chosen_end_old = self.dijkstra_tsp_all_ends(distances_old, start, candidates)
        elif "A*" in algo:
            path_old, distance_old, chosen_end_old = self.astar_tsp_all_ends(distances_old, start, candidates)
        
        if path_old:
            print(f"\n✅ SUCESSO COM ABORDAGEM ANTIGA!")
            print(f"   Caminho: {' → '.join([p.nome for p in path_old])}")
            print(f"   Distância: {distance_old:.2f} pixels")
            
            self.path = path_old
            self.detailed_paths = paths_old
            
            # Gera pontos de desvio APENAS para o caminho final
            self.generate_detour_points_for_final_path()
            
            dist_fmt = self.format_distance(distance_old)
            self.label_distancia.config(text=f"Distância: {dist_fmt}")
            self.label_final.config(text=f"Final escolhido: {chosen_end_old.nome}")
            self.label_strategy.config(text="Estratégia: Linha reta ⚡")
            self.label_desvios.config(text=f"Pontos de desvio: {len(self.points['desvios'])}")
            tempo_total = time.time() - tempo_inicio
            self.label_tempo.config(text=f"Tempo de execução: {tempo_total:.2f}s")
            self.redraw_points()
            self.redraw_path()
            
            ordem = " → ".join([p.nome for p in path_old])
            messagebox.showinfo("Sucesso", 
                              f"Caminho gerado com {algo}!\n\n"
                              f"Estratégia: Linha reta (rápida) ⚡\n\n"
                              f"Final: {chosen_end_old.nome}\n"
                              f"Ordem: {ordem}\n\n"
                              f"Distância: {dist_fmt}\n"
                              f"Pontos de desvio: {len(self.points['desvios'])}\n"
                              f"Tempo de execução: {tempo_total:.2f}s")
            return
        
        print(f"\n❌ Abordagem antiga FALHOU")
        print(f"   Avançando para Nível 2...\n")
        
        # NÍVEL 2: ABORDAGEM HÍBRIDA
        self.label_distancia.config(text="Tentando contornos...")
        self.label_strategy.config(text="Estratégia: Híbrida (A* quando necessário)")
        self.root.update()
        
        print("┌" + "─"*58 + "┐")
        print("│ NÍVEL 2: ABORDAGEM HÍBRIDA (Linha reta + A*)            │")
        print("└" + "─"*58 + "┘")
        
        distances_hybrid, _, paths_hybrid = self.build_complete_graph("hibrido")
        
        path_hybrid = None
        distance_hybrid = float('inf')
        chosen_end_hybrid = None
        
        if "Guloso" in algo:
            path_hybrid, distance_hybrid, chosen_end_hybrid = self.greedy_tsp_all_ends(distances_hybrid, start, candidates)
        elif "Dijkstra" in algo:
            path_hybrid, distance_hybrid, chosen_end_hybrid = self.dijkstra_tsp_all_ends(distances_hybrid, start, candidates)
        elif "A*" in algo:
            path_hybrid, distance_hybrid, chosen_end_hybrid = self.astar_tsp_all_ends(distances_hybrid, start, candidates)
        
        if path_hybrid:
            print(f"\n✅ SUCESSO COM ABORDAGEM HÍBRIDA!")
            print(f"   Caminho: {' → '.join([p.nome for p in path_hybrid])}")
            print(f"   Distância: {distance_hybrid:.2f} pixels")
            
            self.path = path_hybrid
            self.detailed_paths = paths_hybrid
            
            # Gera pontos de desvio APENAS para o caminho final
            self.generate_detour_points_for_final_path()
            
            dist_fmt = self.format_distance(distance_hybrid)
            self.label_distancia.config(text=f"Distância: {dist_fmt}")
            self.label_final.config(text=f"Final escolhido: {chosen_end_hybrid.nome}")
            self.label_strategy.config(text="Estratégia: Híbrida (contornos) 🔄")
            self.label_desvios.config(text=f"Pontos de desvio: {len(self.points['desvios'])}")
            tempo_total = time.time() - tempo_inicio
            self.label_tempo.config(text=f"Tempo de execução: {tempo_total:.2f}s")
            self.redraw_points()
            self.redraw_path()
            
            ordem = " → ".join([p.nome for p in path_hybrid])
            messagebox.showinfo("Sucesso", 
                              f"Caminho gerado com {algo}!\n\n"
                              f"Estratégia: Híbrida (com contornos A*) 🔄\n\n"
                              f"Final: {chosen_end_hybrid.nome}\n"
                              f"Ordem: {ordem}\n\n"
                              f"Distância: {dist_fmt}\n"
                              f"Pontos de desvio: {len(self.points['desvios'])}\n"
                              f"Tempo de execução: {tempo_total:.2f}s")
            return
        
        print(f"\n❌ Abordagem híbrida FALHOU")
        print(f"   Avançando para Nível 3...\n")
        
        # NÍVEL 3: SEMPRE A*
        self.label_distancia.config(text="Tentando força máxima (A*)...")
        self.label_strategy.config(text="Estratégia: Sempre A*")
        self.root.update()
        
        print("┌" + "─"*58 + "┐")
        print("│ NÍVEL 3: SEMPRE A* (Força máxima)                       │")
        print("└" + "─"*58 + "┘")
        
        distances_full, _, paths_full = self.build_complete_graph("sempre_astar")
        
        path_full = None
        distance_full = float('inf')
        chosen_end_full = None
        
        if "Guloso" in algo:
            path_full, distance_full, chosen_end_full = self.greedy_tsp_all_ends(distances_full, start, candidates)
        elif "Dijkstra" in algo:
            path_full, distance_full, chosen_end_full = self.dijkstra_tsp_all_ends(distances_full, start, candidates)
        elif "A*" in algo:
            path_full, distance_full, chosen_end_full = self.astar_tsp_all_ends(distances_full, start, candidates)
        
        if path_full:
            print(f"\n✅ SUCESSO COM SEMPRE A*!")
            print(f"   Caminho: {' → '.join([p.nome for p in path_full])}")
            print(f"   Distância: {distance_full:.2f} pixels")
            
            self.path = path_full
            self.detailed_paths = paths_full
            
            # Gera pontos de desvio APENAS para o caminho final
            self.generate_detour_points_for_final_path()
            
            dist_fmt = self.format_distance(distance_full)
            self.label_distancia.config(text=f"Distância: {dist_fmt}")
            self.label_final.config(text=f"Final escolhido: {chosen_end_full.nome}")
            self.label_strategy.config(text="Estratégia: Sempre A* 🔥")
            self.label_desvios.config(text=f"Pontos de desvio: {len(self.points['desvios'])}")
            tempo_total = time.time() - tempo_inicio
            self.label_tempo.config(text=f"Tempo de execução: {tempo_total:.2f}s")
            self.redraw_points()
            self.redraw_path()
            
            ordem = " → ".join([p.nome for p in path_full])
            messagebox.showinfo("Sucesso", 
                              f"Caminho gerado com {algo}!\n\n"
                              f"Estratégia: Sempre A* (máximo esforço) 🔥\n\n"
                              f"Final: {chosen_end_full.nome}\n"
                              f"Ordem: {ordem}\n\n"
                              f"Distância: {dist_fmt}\n"
                              f"Pontos de desvio: {len(self.points['desvios'])}\n"
                              f"Tempo de execução: {tempo_total:.2f}s")
            return
        
        # FALHOU
        print(f"\n❌ TODAS AS ESTRATÉGIAS FALHARAM!")
        
        self.label_distancia.config(text="Distância: -")
        self.label_final.config(text="Final escolhido: -")
        self.label_strategy.config(text="Estratégia: FALHOU ❌")
        self.label_desvios.config(text="Pontos de desvio: 0")
        self.label_tempo.config(text="Tempo de execução: -")
        
        messagebox.showerror("Erro", "Não foi possível encontrar caminho!")

    def export_to_xml(self):
        if not self.path:
            messagebox.showwarning("Aviso","Nenhum caminho gerado")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".xml", filetypes=[("XML files","*.xml")])
        if not file_path:
            return
        
        root = ET.Element("Pontos")
        
        # Exporta pontos do caminho principal (formato antigo compatível)
        for p in self.path:
            ponto_elem = ET.SubElement(root, "Ponto")
            ponto_elem.set("nome", p.nome)
            ponto_elem.set("lat", str(p.lat))
            ponto_elem.set("lon", str(p.lon))
            ponto_elem.set("tipo", p.tipo)
        
        # Exporta pontos de desvio (no mesmo nível, formato compatível)
        if self.points["desvios"]:
            for p in self.points["desvios"]:
                ponto_elem = ET.SubElement(root, "Ponto")
                ponto_elem.set("nome", p.nome)
                ponto_elem.set("lat", str(p.lat))
                ponto_elem.set("lon", str(p.lon))
                ponto_elem.set("tipo", "desvio")
        
        # Formata e salva
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(file_path, encoding="utf-8", xml_declaration=True)
        
        messagebox.showinfo("Sucesso",
                          f"Arquivo salvo em {file_path}\n\n"
                          f"Pontos do caminho: {len(self.path)}\n"
                          f"Pontos de desvio: {len(self.points['desvios'])}")

    def export_points_only(self):
        """
        Exporta apenas os pontos definidos (início + normais + desvios),
        sem necessidade de rota gerada.
        """
        tem_pontos = self.points["inicio"] or self.points["normais"] or self.points["desvios"]
        if not tem_pontos:
            messagebox.showwarning("Aviso", "Nenhum ponto definido para exportar.\n"
                                            "Adicione ao menos um ponto antes de exportar.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xml",
            filetypes=[("XML files", "*.xml")],
            title="Exportar Pontos"
        )
        if not file_path:
            return

        root = ET.Element("Pontos")

        # Ponto de início
        if self.points["inicio"]:
            p = self.points["inicio"]
            ponto_elem = ET.SubElement(root, "Ponto")
            ponto_elem.set("nome", p.nome)
            ponto_elem.set("lat", str(p.lat))
            ponto_elem.set("lon", str(p.lon))
            ponto_elem.set("tipo", "inicio")

        # Pontos normais
        for p in self.points["normais"]:
            ponto_elem = ET.SubElement(root, "Ponto")
            ponto_elem.set("nome", p.nome)
            ponto_elem.set("lat", str(p.lat))
            ponto_elem.set("lon", str(p.lon))
            ponto_elem.set("tipo", "normal")

        # Pontos de desvio (caso existam)
        for p in self.points["desvios"]:
            ponto_elem = ET.SubElement(root, "Ponto")
            ponto_elem.set("nome", p.nome)
            ponto_elem.set("lat", str(p.lat))
            ponto_elem.set("lon", str(p.lon))
            ponto_elem.set("tipo", "desvio")

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(file_path, encoding="utf-8", xml_declaration=True)

        total = (1 if self.points["inicio"] else 0) + len(self.points["normais"]) + len(self.points["desvios"])
        messagebox.showinfo("Sucesso",
                            f"Pontos exportados em:\n{file_path}\n\n"
                            f"Início: {'Sim' if self.points['inicio'] else 'Não'}\n"
                            f"Pontos normais: {len(self.points['normais'])}\n"
                            f"Pontos de desvio: {len(self.points['desvios'])}\n"
                            f"Total: {total} ponto(s)")

    def import_from_xml(self): 
        file_path = filedialog.askopenfilename(filetypes=[("XML", "*.xml")])
        if not file_path:
            return
        
        tree = ET.parse(file_path)
        root_elem = tree.getroot()
        self.points = {"inicio": None, "normais": [], "desvios": []}
        
        # Detecta formato do XML
        # Formato NOVO: <Rotas><Caminho><Ponto>...
        # Formato ANTIGO: <path><point> ou raiz com <Ponto>
        
        caminho_elem = root_elem.find("Pontos")
        
        if caminho_elem is not None:
            # FORMATO NOVO - com estrutura Rotas/Caminho/PontosDesvio
            print("Detectado formato NOVO de XML")
            
            # Importa pontos do caminho
            for ponto_elem in caminho_elem.findall("Ponto"):
                nome = ponto_elem.get("nome")
                tipo = ponto_elem.get("tipo")
                lat = float(ponto_elem.get("lat"))
                lon = float(ponto_elem.get("lon"))
                
                p = Ponto(nome, lat, lon)
                if self.transform:
                    x, y = self.geo_to_pixel(lon, lat)
                    p.x, p.y = x, y
                
                if tipo == "inicio":
                    p.tipo = "inicio"
                    self.points["inicio"] = p
                else:
                    p.tipo = "normal"
                    self.points["normais"].append(p)
            
            # Importa pontos de desvio (se existirem)
            desvios_elem = root_elem.find("PontosDesvio")
            if desvios_elem is not None:
                for ponto_elem in desvios_elem.findall("Ponto"):
                    nome = ponto_elem.get("nome")
                    lat = float(ponto_elem.get("lat"))
                    lon = float(ponto_elem.get("lon"))
                    
                    p = Ponto(nome, lat, lon)
                    if self.transform:
                        x, y = self.geo_to_pixel(lon, lat)
                        p.x, p.y = x, y
                    p.tipo = "desvio"
                    self.points["desvios"].append(p)
        
        else:
            # FORMATO ANTIGO - estrutura plana com <Ponto> direto na raiz ou <path><point>
            print("Detectado formato ANTIGO de XML")
            
            # Tenta encontrar <point> (formato muito antigo: <path><point>)
            pontos_encontrados = root_elem.findall("point")
            
            # Se não encontrou <point>, procura <Ponto> direto na raiz
            if not pontos_encontrados:
                pontos_encontrados = root_elem.findall("Ponto")
            
            for ponto_elem in pontos_encontrados:
                # Tenta pegar atributos (podem estar como "nome" ou "name", "lat", "lon")
                nome = ponto_elem.get("nome") or ponto_elem.get("name")
                tipo = ponto_elem.get("tipo")
                
                # Tenta pegar lat/lon dos atributos
                lat_str = ponto_elem.get("lat")
                lon_str = ponto_elem.get("lon")
                
                # Se não estão nos atributos, tenta nos sub-elementos
                if lat_str is None:
                    lat_elem = ponto_elem.find("lat")
                    lat_str = lat_elem.text if lat_elem is not None else None
                
                if lon_str is None:
                    lon_elem = ponto_elem.find("lon")
                    lon_str = lon_elem.text if lon_elem is not None else None
                
                if lat_str is None or lon_str is None:
                    print(f"Aviso: Ponto {nome} sem coordenadas, ignorando...")
                    continue
                
                lat = float(lat_str)
                lon = float(lon_str)
                
                p = Ponto(nome, lat, lon)
                if self.transform:
                    x, y = self.geo_to_pixel(lon, lat)
                    p.x, p.y = x, y
                
                # Determina o tipo do ponto
                if nome == "inicio" or tipo == "inicio":
                    p.tipo = "inicio"
                    self.points["inicio"] = p
                else:
                    p.tipo = "normal"
                    self.points["normais"].append(p)

        
        
        # ✅ DEBUG: Verifica pontos fora dos limites da imagem
        if self.original_image:
            img_width = self.original_image.width
            img_height = self.original_image.height
            pontos_fora = []

            todos_pontos = []
            if self.points["inicio"]:
                todos_pontos.append(self.points["inicio"])
            todos_pontos.extend(self.points["normais"])
            todos_pontos.extend(self.points["desvios"])

            for p in todos_pontos:
                x, y = int(p.x), int(p.y)
                fora = []
                if x < 0:          fora.append(f"x={x} (mínimo: 0)")
                if x >= img_width:  fora.append(f"x={x} (máximo: {img_width-1})")
                if y < 0:          fora.append(f"y={y} (mínimo: 0)")
                if y >= img_height: fora.append(f"y={y} (máximo: {img_height-1})")

                if fora:
                    pontos_fora.append((p, fora))

            if pontos_fora:
                print("\n" + "⚠️ "*20)
                print("PONTOS FORA DOS LIMITES DA IMAGEM:")
                print(f"  Imagem: {img_width} x {img_height} pixels")
                print("-" * 60)
                for p, motivos in pontos_fora:
                    print(f"  ❌ {p.nome} (tipo={p.tipo})")
                    print(f"     lat={p.lat}, lon={p.lon}")
                    print(f"     Problemas: {', '.join(motivos)}")
                print("-" * 60)
                print(f"  Total: {len(pontos_fora)} ponto(s) fora dos limites")
                print("⚠️ "*20 + "\n")

                # Mostra aviso na tela também
                nomes = ", ".join([p.nome for p, _ in pontos_fora])
                messagebox.showwarning(
                    "⚠️ Pontos Fora da Imagem",
                    f"{len(pontos_fora)} ponto(s) fora dos limites da imagem:\n\n"
                    f"{nomes}\n\n"
                    f"Imagem: {img_width} x {img_height} px\n\n"
                    "Verifique o console para detalhes.\n"
                    "Esses pontos serão ignorados no cálculo do caminho."
                )
            else:
                print("✅ Todos os pontos estão dentro dos limites da imagem.")
        else:
            print("⚠️ Nenhuma imagem carregada — não foi possível validar os pontos.")

        self.redraw_points()
        self.path = []
        self.detailed_paths = {}
        
        # Mensagem de sucesso
        msg = "Pontos importados com sucesso!\n\n"
        msg += f"Ponto de início: {'Sim' if self.points['inicio'] else 'Não'}\n"
        msg += f"Pontos normais: {len(self.points['normais'])}\n"
        msg += f"Pontos de desvio: {len(self.points['desvios'])}"
        
        messagebox.showinfo("Importado", msg)

# ==============================
# Inicialização
# ==============================
if __name__=="__main__":
    root = tk.Tk()
    app = PathPlannerApp(root)
    root.mainloop()\
    
    
    # Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    #  .\venv\Scripts\activate