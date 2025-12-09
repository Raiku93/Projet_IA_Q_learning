import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import json 
from mpl_toolkits.mplot3d import Axes3D

# --- 1. CONFIGURATION & ESTHÉTIQUE PRO ---
TAILLE_GRILLE_XY = 10 

# Design System "AeroPro"
THEME = {
    # Palette Principale
    "primary": "#2962FF",       # Bleu Royal (Actions principales)
    "primary_dark": "#0039CB",
    "secondary": "#455A64",     # Gris Bleu (Neutre)
    "accent": "#FF6D00",        # Orange Vif (Focus/Action importante)
    
    # Fonds
    "bg_app": "#F5F7FA",        # Gris très très clair (Fond application)
    "bg_sidebar": "#1A2327",    # Noir/Bleu profond (Sidebar)
    "bg_card": "#FFFFFF",       # Blanc (Conteneurs)
    
    # Texte
    "text_main": "#263238",     # Encre sombre
    "text_sidebar": "#ECEFF1",  # Blanc cassé
    "text_muted": "#78909C",    # Gris moyen
    
    # Indicateurs
    "success": "#00C853",       # Vert vibrant
    "warning": "#FFAB00",       # Ambre
    "danger": "#D50000",        # Rouge
    "info": "#00B0FF"           # Cyan
}

# Mapping Physique -> Couleurs UI
COLORS = {
    "obstacle": "#263238",   # Anthracite (Solide)
    "vent": "#90CAF9",       # Bleu très clair (Air)
    "thermique": "#FFAB91",  # Saumon (Chaleur)
    "descendant": "#80DEEA", # Cyan pâle (Froid)
    "inertie": "#CE93D8",    # Mauve (Instabilité)
    "depart": "#43A047",     # Vert (Positif)
    "cible": "#FFD600",      # Or (Objectif)
    "vide": "#FFFFFF",       # Blanc
    "grid": "#ECEFF1"        # Gris ultra léger
}

ACTIONS = {
    0: 'HAUT_2D', 1: 'BAS_2D', 2: 'GAUCHE', 3: 'DROITE',
    4: 'MONTER', 5: 'DESCENDRE'
}
NB_ACTIONS = len(ACTIONS)

# --- Récompenses (Modèle Physique) ---
RECOMPENSE_CIBLE = 1000
RECOMPENSE_COLLISION = -50
RECOMPENSE_MOUVEMENT_2D = -1
RECOMPENSE_MOUVEMENT_3D = -3
RECOMPENSE_PENALITE_VENT = -4
RECOMPENSE_BONUS_FLUX = 5       
RECOMPENSE_MALUS_CONTRE_FLUX = -10 
PROBABILITE_GLISSEMENT = 0.5    

# --- 2. UX Helpers ---
class ToolTip(object):
    """ Gestionnaire de bulles d'aide modernes (Style Dark) """
    def __init__(self, widget, text='Info'):
        self.waittime = 400     #ms
        self.wraplength = 300   #pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 20
        y += self.widget.winfo_rooty() + 25
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        
        # Style Tooltip
        lbl = tk.Label(self.tw, text=self.text, justify='left',
                       background="#263238", foreground="#FFFFFF", 
                       relief='flat', borderwidth=0,
                       wraplength=self.wraplength, 
                       font=("Segoe UI", 9), padx=12, pady=8)
        lbl.pack()

    def hidetip(self):
        tw = self.tw
        self.tw= None
        if tw: tw.destroy()

# --- 3. Moteur IA (Backend) ---
class QLearningParameters:
    def __init__(self, taille_etat_base, nb_actions):
        self.TAILLE_ETAT_BASE = taille_etat_base
        self.NB_ACTIONS = nb_actions
        self.EPISODES = 10000 
        self.ALPHA = 0.1 
        self.GAMMA = 0.9 
        self.EPSILON_INIT = 1.0
        self.DECAY_EPSILON = 0.9996 
        self.MIN_EPSILON = 0.01

    def calculer_episodes_conseilles(self, coords_obstacles):
        nombre_etats_libres = self.TAILLE_ETAT_BASE - len(coords_obstacles)
        facteur_complexite = 1.5 
        conseil = int(nombre_etats_libres * 20 * facteur_complexite)
        return max(5000, min(30000, conseil))

def coords_to_idx(coords, taille_xy):
    r, c, a = coords
    taille_couche_xy = taille_xy * taille_xy
    return a * taille_couche_xy + r * taille_xy + c

def initialiser_positions_vide(taille_xy, niveaux_alt):
    coords_depart = (taille_xy - 1, 0, 0)
    coords_cible = (0, taille_xy - 1, niveaux_alt - 1)
    return coords_depart, coords_cible, [], [], [], [], []

def deplacer_simple(r, c, a, action_index):
    nr, nc, na = r, c, a
    if action_index == 0: nr -= 1    
    elif action_index == 1: nr += 1  
    elif action_index == 2: nc -= 1  
    elif action_index == 3: nc += 1  
    elif action_index == 4: na += 1  
    elif action_index == 5: na -= 1  
    return nr, nc, na

def est_valide(r, c, a, taille_xy, niveaux_alt, obstacles):
    if not (0 <= r < taille_xy and 0 <= c < taille_xy and 0 <= a < niveaux_alt): return False
    if (r, c, a) in obstacles: return False
    return True

def obtenir_etat_recompense_suivants(etat_courant, action_index, coords_cible, coords_obstacles, 
                                     zones_vent, zones_thermiques, zones_descendantes, zones_inertie,
                                     taille_xy, niveaux_alt, mode_stochastique=True):
    r, c, a = etat_courant
    new_r, new_c, new_a = deplacer_simple(r, c, a, action_index)
    
    if not est_valide(new_r, new_c, new_a, taille_xy, niveaux_alt, coords_obstacles):
        return etat_courant, RECOMPENSE_COLLISION, True 

    recompense_mouvement = 0
    etat_intermediaire = (new_r, new_c, new_a)

    if action_index in [0, 1, 2, 3]: recompense_mouvement = RECOMPENSE_MOUVEMENT_2D
    elif action_index in [4, 5]: recompense_mouvement = RECOMPENSE_MOUVEMENT_3D

    # Modélisation des flux
    if etat_intermediaire in zones_vent: recompense_mouvement += RECOMPENSE_PENALITE_VENT

    if etat_courant in zones_thermiques:
        if action_index == 4: recompense_mouvement += RECOMPENSE_BONUS_FLUX 
        elif action_index == 5: recompense_mouvement += RECOMPENSE_MALUS_CONTRE_FLUX 
            
    if etat_courant in zones_descendantes:
        if action_index == 5: recompense_mouvement += RECOMPENSE_BONUS_FLUX
        elif action_index == 4: recompense_mouvement += RECOMPENSE_MALUS_CONTRE_FLUX

    etat_final = etat_intermediaire
    
    # Modélisation du cisaillement (Shear)
    if etat_intermediaire in zones_inertie and action_index in [0, 1, 2, 3]:
        if mode_stochastique and random.random() < PROBABILITE_GLISSEMENT:
            slide_r, slide_c, slide_a = deplacer_simple(new_r, new_c, new_a, action_index)
            if not est_valide(slide_r, slide_c, slide_a, taille_xy, niveaux_alt, coords_obstacles):
                return etat_intermediaire, RECOMPENSE_COLLISION, True
            else:
                etat_final = (slide_r, slide_c, slide_a)

    if etat_final == coords_cible:
        return etat_final, RECOMPENSE_CIBLE + recompense_mouvement, True
    
    return etat_final, recompense_mouvement, False

class QLearningAgent:
    def __init__(self, params: QLearningParameters, taille_xy, niveaux_altitude):
        self.params = params
        self.TAILLE_GRILLE_XY = taille_xy
        self.NIVEAUX_ALTITUDE = niveaux_altitude
        
        self.coords_depart, self.coords_cible, self.coords_obstacles, self.zones_vent, \
        self.zones_thermiques, self.zones_descendantes, self.zones_inertie = \
            initialiser_positions_vide(self.TAILLE_GRILLE_XY, self.NIVEAUX_ALTITUDE)
            
        self.TAILLE_ETAT = self.TAILLE_GRILLE_XY * self.TAILLE_GRILLE_XY * self.NIVEAUX_ALTITUDE
        self.Q_table = np.zeros((self.TAILLE_ETAT, NB_ACTIONS))
        self.historique_recompenses = []
        self.meilleure_recompense = -float('inf')
        self.meilleur_chemin = []
        self.EPSILON = self.params.EPSILON_INIT
        self.MAX_STEPS_EPISODE = 100

    def reset_for_training(self):
        self.historique_recompenses = []
        self.Q_table = np.zeros((self.TAILLE_ETAT, NB_ACTIONS))
        self.EPSILON = self.params.EPSILON_INIT
        self.meilleure_recompense = -float('inf')
        self.meilleur_chemin = []
        etats_libres = self.TAILLE_ETAT - len(self.coords_obstacles)
        self.MAX_STEPS_EPISODE = max(50, int(etats_libres * 2.5))

    def train(self, progress_callback=None, live_path_callback=None):
        self.reset_for_training()
        ALPHA, GAMMA, EPISODES = self.params.ALPHA, self.params.GAMMA, self.params.EPISODES
        path_update_interval = max(1, EPISODES // 100)

        for episode in range(EPISODES):
            curr = self.coords_depart
            if curr is None: break
            
            idx_curr = coords_to_idx(curr, self.TAILLE_GRILLE_XY)
            done = False
            path = [curr]
            ep_reward = 0
            steps = 0
            
            while not done:
                if random.random() < self.EPSILON: action = random.choice(list(ACTIONS.keys()))
                else:
                    max_q = np.max(self.Q_table[idx_curr, :])
                    actions = np.where(self.Q_table[idx_curr, :] == max_q)[0]
                    action = random.choice(actions) if len(actions) > 0 else random.choice(list(ACTIONS.keys()))
                
                next_s, r, done_env = obtenir_etat_recompense_suivants(
                    curr, action, self.coords_cible, self.coords_obstacles, self.zones_vent, 
                    self.zones_thermiques, self.zones_descendantes, self.zones_inertie,
                    self.TAILLE_GRILLE_XY, self.NIVEAUX_ALTITUDE
                )
                
                idx_next = coords_to_idx(next_s, self.TAILLE_GRILLE_XY)
                ep_reward += r
                path.append(next_s)
                
                old_q = self.Q_table[idx_curr, action]
                max_next_q = np.max(self.Q_table[idx_next, :]) if not done_env else 0
                self.Q_table[idx_curr, action] = (1 - ALPHA) * old_q + ALPHA * (r + GAMMA * max_next_q)
                
                curr = next_s
                idx_curr = idx_next
                steps += 1
                
                if steps > self.MAX_STEPS_EPISODE:
                    done = True
                    if curr != self.coords_cible: ep_reward += RECOMPENSE_COLLISION 
                else: done = done_env
            
            if self.EPSILON > self.params.MIN_EPSILON: self.EPSILON *= self.params.DECAY_EPSILON
            self.historique_recompenses.append(ep_reward)
            
            if ep_reward > self.meilleure_recompense and curr == self.coords_cible:
                self.meilleure_recompense = ep_reward
                self.meilleur_chemin = list(path)
            
            if progress_callback and episode % 50 == 0: progress_callback(episode, self.meilleure_recompense)
            if live_path_callback and (episode + 1) % path_update_interval == 0: live_path_callback(path)
                
        return self.Q_table, self.meilleur_chemin, self.meilleure_recompense, self.historique_recompenses

def obtenir_chemin_optimal(Q_table, coords_depart, coords_cible, coords_obstacles, zones_vent, 
                           zones_thermiques, zones_descendantes, zones_inertie, taille_xy, niveaux_alt):
    if coords_depart is None or coords_cible is None: return []
    etats_libres = (taille_xy * taille_xy * niveaux_alt) - len(coords_obstacles)
    max_steps = int(etats_libres * 1.5) 
    curr = coords_depart
    path = [curr]
    steps = 0
    
    while curr != coords_cible and steps < max_steps:
        idx = coords_to_idx(curr, taille_xy)
        max_q = np.max(Q_table[idx, :])
        actions = np.where(Q_table[idx, :] == max_q)[0]
        if len(actions) == 0: break 
        action = random.choice(actions)
        
        next_s, _, done = obtenir_etat_recompense_suivants(
            curr, action, coords_cible, coords_obstacles, zones_vent, 
            zones_thermiques, zones_descendantes, zones_inertie,
            taille_xy, niveaux_alt, mode_stochastique=False
        )
        if done and next_s != coords_cible: break
        path.append(next_s)
        curr = next_s
        steps += 1
    return path

# --- 4. Interface Graphique Pro (GUI) ---
class QLearningGUI:
    def __init__(self, master, agent, params):
        self.master = master
        self.agent = agent
        self.params = params
        
        # Configuration Fenêtre
        master.title("NavDrone AI™ [Pro Suite]")
        master.geometry("1400x900")
        master.state('zoomed') # Plein écran par défaut (Windows)
        master.configure(bg=THEME["bg_app"])
        
        # Configuration Matplotlib Global Style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'axes.edgecolor': '#E0E0E0',
            'axes.linewidth': 1,
            'xtick.color': '#757575',
            'ytick.color': '#757575',
            'text.color': '#424242',
            'axes.labelcolor': '#424242',
            'font.family': 'sans-serif',
            'font.sans-serif': ['Segoe UI', 'Arial'],
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'grid.color': '#EEEEEE'
        })

        # Style TTK Avancé
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Définitions de Styles
        self.style.configure("TFrame", background=THEME["bg_app"])
        self.style.configure("Card.TFrame", background=THEME["bg_card"], relief="flat")
        
        # Header Styles
        self.style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"), foreground=THEME["text_main"], background=THEME["bg_app"])
        self.style.configure("SubHeader.TLabel", font=("Segoe UI", 12), foreground=THEME["text_muted"], background=THEME["bg_card"])
        self.style.configure("CardTitle.TLabel", font=("Segoe UI", 11, "bold"), foreground=THEME["primary"], background=THEME["bg_card"])
        
        # Sidebar Styles
        self.style.configure("Sidebar.TFrame", background=THEME["bg_sidebar"])
        self.style.configure("SidebarTitle.TLabel", font=("Segoe UI", 10, "bold", "uppercase"), foreground=THEME["text_muted"], background=THEME["bg_sidebar"])
        self.style.configure("SidebarText.TLabel", font=("Segoe UI", 9), foreground=THEME["text_sidebar"], background=THEME["bg_sidebar"])
        
        # Buttons
        self.style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), background=THEME["primary"], foreground="white", borderwidth=0, focuscolor="none")
        self.style.map("Primary.TButton", background=[('active', THEME["primary_dark"])])
        
        self.style.configure("Success.TButton", font=("Segoe UI", 10, "bold"), background=THEME["success"], foreground="white", borderwidth=0)
        self.style.map("Success.TButton", background=[('active', '#009624')])

        # Notebook (Tabs) style "Navigation"
        self.style.configure("TNotebook", background=THEME["bg_app"], borderwidth=0)
        self.style.configure("TNotebook.Tab", padding=[25, 12], font=("Segoe UI", 11, "bold"), background=THEME["bg_app"], foreground=THEME["text_muted"], borderwidth=0)
        self.style.map("TNotebook.Tab", 
                       background=[("selected", THEME["bg_app"]), ("active", THEME["bg_app"])], 
                       foreground=[("selected", THEME["primary"]), ("active", THEME["text_main"])])

        # Variables d'état
        self.CELL_SIZE = 55 
        self.canvas_width = self.CELL_SIZE * self.agent.TAILLE_GRILLE_XY
        self.canvas_height = self.CELL_SIZE * self.agent.TAILLE_GRILLE_XY
        self.current_path = []
        self.current_step = 0
        self.replay_running = False
        self.training_in_progress = False
        self.current_altitude_level = tk.IntVar(value=0)
        self.edit_mode = tk.StringVar(value="obstacle")
        self.view_mode = tk.StringVar(value="terrain") # "terrain" or "heatmap"
        
        self.setup_layout()
        
    def setup_layout(self):
        # 1. Top Bar (Logo & Global Actions)
        top_bar = ttk.Frame(self.master, height=60, padding="20 10")
        top_bar.pack(fill='x')
        
        # Logo placeholder
        lbl_logo = ttk.Label(top_bar, text="🚁 NavDrone AI", style="Header.TLabel")
        lbl_logo.pack(side=tk.LEFT)
        
        ttk.Label(top_bar, text=" |  Suite de Simulation Autonome v2.4", font=("Segoe UI", 12), foreground=THEME["text_muted"]).pack(side=tk.LEFT, padx=10, pady=4)

        btn_help = ttk.Button(top_bar, text="Documentation", command=self.show_help)
        btn_help.pack(side=tk.RIGHT)

        # 2. Main Content Area
        main_container = ttk.Frame(self.master)
        main_container.pack(fill='both', expand=True, padx=20, pady=(0, 20))

        # Notebook Navigation
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill='both', expand=True)

        # Tabs Creation
        self.frame_editor = ttk.Frame(self.notebook)
        self.frame_3d = ttk.Frame(self.notebook)
        self.frame_stats = ttk.Frame(self.notebook)
        self.frame_params = ttk.Frame(self.notebook)

        self.notebook.add(self.frame_editor, text="DESIGN & SIMULATION")
        self.notebook.add(self.frame_3d, text="VISUALISATION 3D")
        self.notebook.add(self.frame_stats, text="ANALYSE TÉLÉMÉTRIE")
        self.notebook.add(self.frame_params, text="CONFIGURATION SYSTÈME")

        self.setup_editor_tab(self.frame_editor)
        self.setup_3d_frame(self.frame_3d)
        self.setup_stats_tab(self.frame_stats)
        self.setup_params_tab(self.frame_params)

        self.draw_grid()
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

    # --- EDITOR TAB (Layout "Dashboard") ---
    def setup_editor_tab(self, parent):
        # Layout: Left Canvas (Flexible) | Right Sidebar (Fixed width)
        
        # --- LEFT: MAP CONTAINER ---
        map_area = ttk.Frame(parent)
        map_area.pack(side=tk.LEFT, fill='both', expand=True, padx=(0, 20), pady=20)
        
        # Card Wrapper for Map
        map_card = ttk.Frame(map_area, style="Card.TFrame", padding=20)
        map_card.pack(fill='both', expand=True)
        
        # Map Header inside Card
        header_frame = ttk.Frame(map_card, style="Card.TFrame")
        header_frame.pack(fill='x', pady=(0, 20))
        
        self.lbl_etage = ttk.Label(header_frame, text="ALTITUDE (AGL) : 0", font=("Segoe UI", 16, "bold"), foreground=THEME["primary"], background=THEME["bg_card"])
        self.lbl_etage.pack(side=tk.LEFT)
        
        ttk.Label(header_frame, text="Vue Tactique", style="SubHeader.TLabel").pack(side=tk.RIGHT)

        # Canvas Wrapper (For Centering)
        canvas_wrapper = tk.Frame(map_card, bg=THEME["bg_card"])
        canvas_wrapper.pack(fill='both', expand=True)
        
        self.canvas_grid = tk.Canvas(canvas_wrapper, width=self.canvas_width, height=self.canvas_height, 
                                     bg="white", highlightthickness=0)
        self.canvas_grid.pack(anchor="center", expand=True) # Center in available space
        self.canvas_grid.bind("<Button-1>", self.on_grid_click)
        
        # Subtle shadow effect via simpler approach (border)
        self.canvas_grid.config(bd=1, relief="solid")


        # --- RIGHT: CONTROL SIDEBAR ---
        sidebar = ttk.Frame(parent, width=380, style="Sidebar.TFrame")
        sidebar.pack(side=tk.RIGHT, fill='y', pady=0)
        sidebar.pack_propagate(False) # Force width
        
        content_sidebar = ttk.Frame(sidebar, style="Sidebar.TFrame", padding=25)
        content_sidebar.pack(fill='both', expand=True)
        
        # 0. View Mode Switcher
        ttk.Label(content_sidebar, text="MODE VISUALISATION", style="SidebarTitle.TLabel").pack(anchor='w', pady=(0, 10))
        
        view_frame = tk.Frame(content_sidebar, bg=THEME["bg_sidebar"])
        view_frame.pack(fill='x', pady=(0, 20))
        
        # Custom "Tabs" using Radiobuttons
        def create_view_btn(txt, val, col_idx):
            rb = tk.Radiobutton(view_frame, text=txt, variable=self.view_mode, value=val,
                           indicatoron=0, width=15, height=2,
                           bg="#37474F", fg="white", selectcolor=THEME["accent"], 
                           activebackground=THEME["accent"], activeforeground="white",
                           font=("Segoe UI", 9, "bold"), relief="flat", bd=0, cursor="hand2",
                           command=lambda: self.draw_grid(self.current_path))
            rb.grid(row=0, column=col_idx, padx=2, sticky="ew")
        
        view_frame.columnconfigure(0, weight=1)
        view_frame.columnconfigure(1, weight=1)
        create_view_btn("TACTIQUE", "terrain", 0)
        create_view_btn("Cerveau IA", "heatmap", 1)


        # 1. Status Section
        ttk.Label(content_sidebar, text="STATUT MISSION", style="SidebarTitle.TLabel").pack(anchor='w', pady=(0, 10))
        self.lbl_status = ttk.Label(content_sidebar, text="PRÊT", font=("Segoe UI", 12, "bold"), foreground=THEME["success"], background=THEME["bg_sidebar"])
        self.lbl_status.pack(anchor='w', pady=(0, 5))
        
        self.progress = ttk.Progressbar(content_sidebar, length=100, mode='determinate', style="Horizontal.TProgressbar")
        self.progress.pack(fill='x', pady=(0, 20))

        # 2. Layer Control
        ttk.Label(content_sidebar, text="CONTRÔLE ALTITUDE", style="SidebarTitle.TLabel").pack(anchor='w', pady=(10, 10))
        
        scale_style = ttk.Style()
        scale_style.configure("Dark.Horizontal.TScale", background=THEME["bg_sidebar"], troughcolor="#37474F", sliderlength=20)
        self.scale_alt = tk.Scale(content_sidebar, from_=0, to=self.agent.NIVEAUX_ALTITUDE-1, orient=tk.HORIZONTAL, 
                                   variable=self.current_altitude_level, command=self.on_altitude_change,
                                   bg=THEME["bg_sidebar"], fg="white", highlightthickness=0, 
                                   activebackground=THEME["primary"], troughcolor="#263238", bd=0)
        self.scale_alt.pack(fill='x', pady=5)


        # 3. Toolbox Grid
        ttk.Label(content_sidebar, text="BOÎTE À OUTILS", style="SidebarTitle.TLabel").pack(anchor='w', pady=(30, 15))
        
        tools_frame = tk.Frame(content_sidebar, bg=THEME["bg_sidebar"])
        tools_frame.pack(fill='x')
        tools_frame.columnconfigure(0, weight=1)
        tools_frame.columnconfigure(1, weight=1)

        tools = [
            ("Départ", "depart", COLORS["depart"], "🛫"),   
            ("Cible", "cible", COLORS["cible"], "🎯"),
            ("Structure", "obstacle", COLORS["obstacle"], "⬛"),
            ("Gomme", "effacer", "#90A4AE", "🧹"),
            ("Turbulence", "vent", COLORS["vent"], "〰"),         
            ("Ascendance", "thermique", COLORS["thermique"], "🔥"), 
            ("Rabattant", "descendant", COLORS["descendant"], "❄️"),
            ("Cisaillement", "inertie", COLORS["inertie"], "⚡")   
        ]

        for i, (name, mode, col, icon) in enumerate(tools):
            row, col_idx = divmod(i, 2)
            # Custom styled radio button looking like a tile
            rb = tk.Radiobutton(tools_frame, text=f"{icon} {name}", variable=self.edit_mode, value=mode,
                                indicatoron=0, width=12, height=2,
                                bg="#37474F", fg="white", selectcolor=THEME["primary"], 
                                activebackground=THEME["primary_dark"], activeforeground="white",
                                font=("Segoe UI", 10), relief="flat", bd=0, cursor="hand2")
            rb.grid(row=row, column=col_idx, padx=4, pady=4, sticky="ew")
            
            # Contextual Help
            desc = {
                "depart": "Point d'insertion (Start)", "cible": "Objectif (Target)",
                "obstacle": "Zone interdite (Mur)", "effacer": "Supprimer entité",
                "vent": "Zone de traînée (-Score)", "thermique": "Gain altitude (+Score)",
                "descendant": "Perte altitude (+Score)", "inertie": "Risque dérive latérale"
            }
            ToolTip(rb, desc.get(mode, ""))


        # 4. Actions
        btn_frame = tk.Frame(content_sidebar, bg=THEME["bg_sidebar"])
        btn_frame.pack(fill='x', side=tk.BOTTOM, pady=20)

        self.btn_train = ttk.Button(btn_frame, text="▶  LANCER SIMULATION", style="Success.TButton", command=self.start_training_manually)
        self.btn_train.pack(fill='x', pady=5)

        self.btn_replay = ttk.Button(btn_frame, text="↺  REJOUER SCÉNARIO", style="Primary.TButton", command=self.start_replay, state=tk.DISABLED)
        self.btn_replay.pack(fill='x', pady=5)
        
        # File ops small buttons
        file_frame = tk.Frame(btn_frame, bg=THEME["bg_sidebar"])
        file_frame.pack(fill='x', pady=10)
        tk.Button(file_frame, text="Export JSON", bg="#455A64", fg="white", relief="flat", command=self.export_map).pack(side=tk.LEFT, expand=True, fill='x', padx=(0,2))
        tk.Button(file_frame, text="Import JSON", bg="#455A64", fg="white", relief="flat", command=self.import_map).pack(side=tk.RIGHT, expand=True, fill='x', padx=(2,0))


    # --- 3D TAB ---
    def setup_3d_frame(self, frame):
        container = ttk.Frame(frame, style="Card.TFrame", padding=30)
        container.pack(fill='both', expand=True, padx=40, pady=40)
        
        header = ttk.Frame(container, style="Card.TFrame")
        header.pack(fill='x', pady=(0, 10))
        ttk.Label(header, text="Visualisation Volumétrique", style="CardTitle.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, text="Utilisez la souris pour pivoter la vue", style="SubHeader.TLabel", font=("Segoe UI", 9)).pack(side=tk.RIGHT)

        self.fig_3d = plt.figure(figsize=(5, 4))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=container)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.draw_3d_environment([])

    # --- STATS TAB ---
    def setup_stats_tab(self, frame):
        # Grid layout for charts
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)
        
        # Chart 1
        c1 = ttk.Frame(frame, style="Card.TFrame", padding=20)
        c1.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=20)
        ttk.Label(c1, text="Convergence de l'Apprentissage", style="CardTitle.TLabel").pack(anchor='w', pady=(0, 15))
        
        self.fig_conv, self.ax_conv = plt.subplots(figsize=(4,3))
        self.canvas_chart = FigureCanvasTkAgg(self.fig_conv, master=c1)
        self.canvas_chart.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Chart 2
        c2 = ttk.Frame(frame, style="Card.TFrame", padding=20)
        c2.grid(row=0, column=1, sticky="nsew", padx=(10, 20), pady=20)
        ttk.Label(c2, text="Profil de Mission (Z & Score)", style="CardTitle.TLabel").pack(anchor='w', pady=(0, 15))
        
        self.fig_stats, (self.ax_alt, self.ax_reward) = plt.subplots(2, 1, figsize=(4, 4), sharex=True)
        self.canvas_stats = FigureCanvasTkAgg(self.fig_stats, master=c2)
        self.canvas_stats.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- PARAMS TAB ---
    def setup_params_tab(self, frame):
        center_card = ttk.Frame(frame, style="Card.TFrame", padding=40)
        center_card.pack(fill='both', expand=True, padx=100, pady=40)
        
        ttk.Label(center_card, text="Configuration Système", style="CardTitle.TLabel", font=("Segoe UI", 16)).pack(anchor='w', pady=(0, 30))
        
        grid_f = ttk.Frame(center_card, style="Card.TFrame")
        grid_f.pack(fill='x')
        
        def create_row(row, label, var, help_text):
            # Label
            ttk.Label(grid_f, text=label, style="CardTitle.TLabel", font=("Segoe UI", 10)).grid(row=row, column=0, sticky='w', pady=15)
            # Entry
            entry = ttk.Entry(grid_f, textvariable=var, width=15, font=("Segoe UI", 10))
            entry.grid(row=row, column=1, sticky='w', padx=20)
            # Help
            ttk.Label(grid_f, text=help_text, background=THEME["bg_card"], foreground=THEME["text_muted"]).grid(row=row, column=2, sticky='w')

        self.var_episodes = tk.IntVar(value=self.params.EPISODES)
        self.var_alpha = tk.DoubleVar(value=self.params.ALPHA)
        self.var_gamma = tk.DoubleVar(value=self.params.GAMMA)
        self.var_epsilon = tk.DoubleVar(value=self.params.EPSILON_INIT)
        self.var_altitude_max = tk.IntVar(value=self.agent.NIVEAUX_ALTITUDE)
        self.var_taille_xy = tk.IntVar(value=self.agent.TAILLE_GRILLE_XY)

        create_row(0, "Itérations (Epochs)", self.var_episodes, "Cycles d'apprentissage total.")
        create_row(1, "Learning Rate (α)", self.var_alpha, "Vitesse d'adaptation (0.0 - 1.0).")
        create_row(2, "Discount Factor (γ)", self.var_gamma, "Poids du futur (0.0 - 1.0).")
        create_row(3, "Exploration (ε)", self.var_epsilon, "Taux d'aléatoire initial.")
        
        ttk.Separator(grid_f, orient='horizontal').grid(row=4, column=0, columnspan=3, sticky='ew', pady=25)
        
        create_row(5, "Grid Size (X/Y)", self.var_taille_xy, "Largeur de la zone (Min 5, Max 20).")
        create_row(6, "Grid Layers (Z)", self.var_altitude_max, "Couches d'altitude (Min 1, Max 10).")

        btn = ttk.Button(center_card, text="Appliquer les modifications", style="Primary.TButton", command=self.apply_params)
        btn.pack(pady=30, anchor='w')

    # --- LOGIQUE INTERFACE (Canvas & Charts) ---
    def on_tab_change(self, event):
        if self.notebook.index(self.notebook.select()) == 1: 
            self.draw_3d_environment(getattr(self, 'final_optimal_path', []))

    def on_grid_click(self, event):
        if self.training_in_progress: return
        c, r = event.x // self.CELL_SIZE, event.y // self.CELL_SIZE
        a = self.current_altitude_level.get()
        
        if not (0 <= c < self.agent.TAILLE_GRILLE_XY and 0 <= r < self.agent.TAILLE_GRILLE_XY): return
        
        coords = (r, c, a)
        mode = self.edit_mode.get()
        
        lists_to_check = [self.agent.coords_obstacles, self.agent.zones_vent, self.agent.zones_thermiques, 
                          self.agent.zones_descendantes, self.agent.zones_inertie]
        for lst in lists_to_check:
            if coords in lst: lst.remove(coords)
            
        if coords == self.agent.coords_depart and mode != "depart": self.agent.coords_depart = None
        if coords == self.agent.coords_cible and mode != "cible": self.agent.coords_cible = None

        if mode == "depart": self.agent.coords_depart = coords
        elif mode == "cible": self.agent.coords_cible = coords
        elif mode == "obstacle": self.agent.coords_obstacles.append(coords)
        elif mode == "vent": self.agent.zones_vent.append(coords)
        elif mode == "thermique": self.agent.zones_thermiques.append(coords)
        elif mode == "descendant": self.agent.zones_descendantes.append(coords)
        elif mode == "inertie": self.agent.zones_inertie.append(coords)
        
        self.draw_grid()

    def on_altitude_change(self, val):
        val_int = int(float(val))
        self.current_altitude_level.set(val_int)
        self.lbl_etage.config(text=f"ALTITUDE (AGL) : {val_int}")
        path_to_draw = self.current_path if self.training_in_progress else getattr(self, 'final_optimal_path', None)
        self.draw_grid(path_to_draw)

    def draw_grid(self, path=None):
        self.canvas_grid.delete("all")
        sz = self.CELL_SIZE
        view_a = self.current_altitude_level.get()
        is_heatmap = (self.view_mode.get() == "heatmap")
        
        # --- 1. Rendu Heatmap (Valeurs Q) ---
        if is_heatmap:
            # Calculer min/max pour la normalisation de couleur
            # On prend la Q-Table globale pour que les couleurs soient cohérentes
            q_min, q_max = np.min(self.agent.Q_table), np.max(self.agent.Q_table)
            
            # Éviter division par zero si tout est à 0
            if q_max == q_min: q_max += 1

            norm = mcolors.Normalize(vmin=q_min, vmax=q_max)
            cmap = cm.get_cmap('RdYlGn') # Rouge -> Jaune -> Vert

            for r in range(self.agent.TAILLE_GRILLE_XY):
                for c in range(self.agent.TAILLE_GRILLE_XY):
                    idx = coords_to_idx((r, c, view_a), self.agent.TAILLE_GRILLE_XY)
                    # Valeur maximale de l'état (le meilleur coup possible)
                    val = np.max(self.agent.Q_table[idx])
                    
                    color_rgba = cmap(norm(val))
                    color_hex = mcolors.to_hex(color_rgba)
                    
                    x, y = c*sz, r*sz
                    self.canvas_grid.create_rectangle(x, y, x+sz, y+sz, fill=color_hex, outline="")
                    
                    # Debug text (optionnel, peut surcharger)
                    # self.canvas_grid.create_text(x+sz/2, y+sz/2, text=f"{int(val)}", font=('Arial', 8))

        else:
            # --- 1. Rendu Terrain (Classique) ---
            # Fond blanc pur pour "Carte propre"
            self.canvas_grid.create_rectangle(0,0, self.canvas_width, self.canvas_height, fill="white", outline="white")

        # --- 2. Grille ---
        for i in range(self.agent.TAILLE_GRILLE_XY + 1):
            line_col = "#555555" if is_heatmap else COLORS["grid"] # Grille plus sombre en heatmap
            self.canvas_grid.create_line(0, i*sz, self.canvas_width, i*sz, fill=line_col, width=1)
            self.canvas_grid.create_line(i*sz, 0, i*sz, self.canvas_height, fill=line_col, width=1)

        def draw_cell_icon(r, c, color, text="", text_color="white", fill_bg=True):
            x, y = c*sz, r*sz
            # En mode heatmap, on ne remplit pas le fond sauf pour les obstacles majeurs
            if fill_bg and (not is_heatmap or color == COLORS["obstacle"]):
                self.canvas_grid.create_rectangle(x+1, y+1, x+sz-1, y+sz-1, fill=color, outline="")
            
            if text:
                font_size = 18 if sz >= 50 else 10
                # En heatmap, texte noir pour contraste
                tc = "black" if is_heatmap and color != COLORS["obstacle"] else text_color
                self.canvas_grid.create_text(x+sz/2, y+sz/2, text=text, fill=tc, font=('Segoe UI Emoji', font_size))

        # Dessin des entités
        for r in range(self.agent.TAILLE_GRILLE_XY):
            for c in range(self.agent.TAILLE_GRILLE_XY):
                coord = (r, c, view_a)
                
                # En Heatmap, on n'affiche que les murs et cibles/départs pour ne pas surcharger
                # En Terrain, on affiche tout
                
                if coord in self.agent.coords_obstacles: draw_cell_icon(r, c, COLORS["obstacle"])
                
                elif coord in self.agent.zones_vent: 
                     draw_cell_icon(r, c, COLORS["vent"], "〰", fill_bg=not is_heatmap)
                elif coord in self.agent.zones_thermiques: 
                     draw_cell_icon(r, c, COLORS["thermique"], "🔥", fill_bg=not is_heatmap)
                elif coord in self.agent.zones_descendantes: 
                     draw_cell_icon(r, c, COLORS["descendant"], "❄️", fill_bg=not is_heatmap)
                elif coord in self.agent.zones_inertie: 
                     draw_cell_icon(r, c, COLORS["inertie"], "⚡", fill_bg=not is_heatmap)
                
                if coord == self.agent.coords_depart: draw_cell_icon(r, c, COLORS["depart"], "🛫")
                if coord == self.agent.coords_cible: draw_cell_icon(r, c, COLORS["cible"], "🎯")

        # Trajectoire (Ligne lisse avec marqueurs)
        if path:
            points_current_layer = []
            for i, (r, c, a) in enumerate(path):
                if a == view_a:
                    cx, cy = c*sz+sz/2, r*sz+sz/2
                    # En heatmap, on met la trajectoire en Blanc ou Noir pour contraste
                    col_line = "white" if is_heatmap else THEME["primary"]
                    col_dot = "white" if is_heatmap else THEME["accent"]
                    
                    if i==len(path)-1: col_dot = "#FFD600" # Target reach color

                    # Drone marker
                    self.canvas_grid.create_oval(cx-6, cy-6, cx+6, cy+6, fill=col_dot, outline="black", width=1)
                    
                    # Ligne de connexion
                    if i > 0:
                        pr, pc, pa = path[i-1]
                        if pa == a:
                            px, py = pc*sz+sz/2, pr*sz+sz/2
                            self.canvas_grid.create_line(px, py, cx, cy, fill=col_line, width=3, capstyle=tk.ROUND, smooth=True)
                        else:
                            # Indicateur de transition Z
                            txt = "▲" if pa < a else "▼"
                            color_trans = "black" if is_heatmap else "#AA00FF"
                            offset_y = -18 if pa < a else 18
                            self.canvas_grid.create_text(cx+18, cy+offset_y, text=txt, fill=color_trans, font=('Arial', 14, 'bold'))

    def draw_3d_environment(self, path=None):
        if not self.master.winfo_exists(): return
        self.ax_3d.clear()
        self.ax_3d.view_init(elev=35, azim=-50)
        
        # Style 3D Pro
        self.ax_3d.set_xlabel("X (Est)", fontsize=9, labelpad=5)
        self.ax_3d.set_ylabel("Y (Nord)", fontsize=9, labelpad=5)
        self.ax_3d.set_zlabel("Z (Alt)", fontsize=9, labelpad=5)
        
        # Grille 3D minimaliste
        self.ax_3d.xaxis.pane.fill = False
        self.ax_3d.yaxis.pane.fill = False
        self.ax_3d.zaxis.pane.fill = False
        self.ax_3d.xaxis.pane.set_edgecolor('w')
        self.ax_3d.yaxis.pane.set_edgecolor('w')
        self.ax_3d.zaxis.pane.set_edgecolor('w')
        self.ax_3d.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        max_c = self.agent.TAILLE_GRILLE_XY - 1
        max_a = self.agent.NIVEAUX_ALTITUDE - 1
        self.ax_3d.set_xlim(0, max_c+1)
        self.ax_3d.set_ylim(0, max_c+1)
        self.ax_3d.set_zlim(0, max_a+1)
        self.ax_3d.invert_yaxis()

        def draw_voxels(coord_list, color, alpha):
            if not coord_list: return
            for r, c, a in coord_list:
                self.ax_3d.bar3d(c, r, a, 1, 1, 1, color=color, alpha=alpha, shade=True, edgecolor=None)

        draw_voxels(self.agent.coords_obstacles, COLORS["obstacle"], 0.8)
        draw_voxels(self.agent.zones_thermiques, COLORS["thermique"], 0.3)
        draw_voxels(self.agent.zones_descendantes, COLORS["descendant"], 0.3)
        draw_voxels(self.agent.zones_vent, COLORS["vent"], 0.2)
        draw_voxels(self.agent.zones_inertie, COLORS["inertie"], 0.2)

        if self.agent.coords_depart:
            d = self.agent.coords_depart
            self.ax_3d.scatter(d[1]+0.5, d[0]+0.5, d[2]+0.5, c=COLORS["depart"], s=200, label="Départ", edgecolors='white', alpha=1)
        if self.agent.coords_cible:
            t = self.agent.coords_cible
            self.ax_3d.scatter(t[1]+0.5, t[0]+0.5, t[2]+0.5, c=COLORS["cible"], marker='*', s=400, label="Cible", edgecolors='black', alpha=1)

        if path:
            xs = [p[1]+0.5 for p in path]
            ys = [p[0]+0.5 for p in path]
            zs = [p[2]+0.5 for p in path]
            self.ax_3d.plot(xs, ys, zs, c=THEME["primary"], linewidth=3, marker='o', markersize=4, alpha=0.8)
        
        self.canvas_3d.draw()

    # --- LOGIQUE CONTROLE ---
    def start_training_manually(self):
        if not self.agent.coords_depart or not self.agent.coords_cible:
            messagebox.showwarning("Configuration Incomplète", "Veuillez définir un point de Départ (🛫) et une Cible (🎯).")
            return
        
        self.training_in_progress = True
        self.btn_train.config(state=tk.DISABLED)
        self.btn_replay.config(state=tk.DISABLED)
        self.lbl_status.config(text="CALCUL EN COURS...", foreground=THEME["warning"])
        self.progress['value'] = 0
        
        threading.Thread(target=self.run_training, daemon=True).start()

    def run_training(self):
        self.agent.train(
            progress_callback=lambda ep, rew: self.master.after(0, self.update_progress, ep, rew),
            live_path_callback=lambda p: self.master.after(0, self.update_live, p)
        )
        self.master.after(0, self.finish_training)

    def update_progress(self, ep, rew):
        self.progress['value'] = (ep / self.params.EPISODES) * 100
        # Texte de statut dynamique
        self.lbl_status.config(text=f"OPTIMISATION... {int((ep/self.params.EPISODES)*100)}%")

    def update_live(self, path):
        self.current_path = path
        if self.notebook.index(self.notebook.select()) == 0: 
            if path and path[-1][2] != self.current_altitude_level.get():
                self.current_altitude_level.set(path[-1][2])
                self.lbl_etage.config(text=f"ALTITUDE (AGL) : {path[-1][2]}")
            self.draw_grid(path)

    def finish_training(self):
        self.training_in_progress = False
        self.btn_train.config(state=tk.NORMAL)
        self.btn_replay.config(state=tk.NORMAL)
        self.lbl_status.config(text="MISSION OPTIMISÉE", foreground=THEME["success"])
        self.progress['value'] = 100
        
        self.final_optimal_path = obtenir_chemin_optimal(
            self.agent.Q_table, self.agent.coords_depart, self.agent.coords_cible,
            self.agent.coords_obstacles, self.agent.zones_vent,
            self.agent.zones_thermiques, self.agent.zones_descendantes, self.agent.zones_inertie,
            self.agent.TAILLE_GRILLE_XY, self.agent.NIVEAUX_ALTITUDE
        )
        
        self.draw_grid(self.final_optimal_path)
        self.plot_graphs()
        messagebox.showinfo("Succès", "Calcul de trajectoire terminé.\nVisualisez les résultats dans l'onglet 'Visualisation 3D'.")

    def start_replay(self):
        if not hasattr(self, 'final_optimal_path') or not self.final_optimal_path: return
        self.replay_running = True
        self.current_path = self.final_optimal_path
        self.current_step = 0
        self.btn_replay.config(state=tk.DISABLED)
        self.loop_replay()

    def loop_replay(self):
        if not self.replay_running: return
        if self.current_step < len(self.current_path):
            p = self.current_path[self.current_step]
            if p[2] != self.current_altitude_level.get(): 
                self.current_altitude_level.set(p[2])
                self.lbl_etage.config(text=f"ALTITUDE (AGL) : {p[2]}")
            
            self.draw_grid(self.current_path[:self.current_step+1])
            self.current_step += 1
            self.master.after(150, self.loop_replay)
        else:
            self.replay_running = False
            self.btn_replay.config(state=tk.NORMAL)

    def plot_graphs(self):
        # 1. Convergence Style Pro
        self.ax_conv.clear()
        if self.agent.historique_recompenses:
            # Ligne brute très légère
            self.ax_conv.plot(self.agent.historique_recompenses, color=THEME["text_muted"], alpha=0.2, linewidth=0.5, label="Raw")
            # Moyenne glissante
            w = 50
            if len(self.agent.historique_recompenses) > w:
                avg = np.convolve(self.agent.historique_recompenses, np.ones(w)/w, mode='valid')
                self.ax_conv.plot(np.arange(w-1, len(self.agent.historique_recompenses)), avg, color=THEME["primary"], label="Tendance (SMA)", linewidth=2)
        
        self.ax_conv.set_title("Performance d'Apprentissage", fontsize=10, fontweight='bold', pad=10)
        self.ax_conv.set_xlabel("Épisodes", fontsize=8)
        self.ax_conv.set_ylabel("Reward", fontsize=8)
        self.ax_conv.tick_params(labelsize=8)
        self.ax_conv.grid(True, linestyle=':', alpha=0.6)
        self.canvas_chart.draw()
        
        # 2. Profil de Vol
        p = self.final_optimal_path
        self.ax_alt.clear(); self.ax_reward.clear()
        if p and len(p) > 1:
            steps = range(len(p))
            # Altitude : Step plot rempli
            self.ax_alt.fill_between(steps, [x[2] for x in p], color=THEME["primary"], alpha=0.1)
            self.ax_alt.step(steps, [x[2] for x in p], color=THEME["primary"], where='mid', linewidth=2)
            self.ax_alt.set_ylabel("Altitude Z", fontsize=8)
            self.ax_alt.set_title("Profil Vertical de Mission", fontsize=10, fontweight='bold')
            self.ax_alt.grid(True, linestyle=':', alpha=0.6)
            
            # Score cumulé
            rewards = []
            cum = 0
            for i in range(len(p)-1):
                curr, nxt = p[i], p[i+1]
                r_step = -1 
                if nxt == self.agent.coords_cible: r_step += 1000
                cum += r_step
                rewards.append(cum)
            
            self.ax_reward.plot(rewards, color=THEME["success"], linewidth=2)
            self.ax_reward.set_ylabel("Score Cumulé", fontsize=8)
            self.ax_reward.set_xlabel("Points de Navigation (Waypoints)", fontsize=8)
            self.ax_reward.grid(True, linestyle=':', alpha=0.6)

        self.fig_stats.tight_layout()
        self.canvas_stats.draw()

    def apply_params(self):
        if self.training_in_progress: 
            messagebox.showwarning("Attention", "Impossible de modifier les paramètres pendant la simulation.")
            return

        # Application
        self.params.EPISODES = self.var_episodes.get()
        self.params.ALPHA = self.var_alpha.get()
        self.params.GAMMA = self.var_gamma.get()
        self.params.EPSILON_INIT = self.var_epsilon.get()

        new_xy = max(5, min(20, self.var_taille_xy.get()))
        new_alt = max(1, min(10, self.var_altitude_max.get()))
        
        if new_xy != self.agent.TAILLE_GRILLE_XY or new_alt != self.agent.NIVEAUX_ALTITUDE:
            self.agent.TAILLE_GRILLE_XY = new_xy
            self.agent.NIVEAUX_ALTITUDE = new_alt
            self.agent.TAILLE_ETAT = new_xy * new_xy * new_alt
            self.agent.reset_for_training()
            
            # Filtre objets hors zone
            features = [self.agent.coords_obstacles, self.agent.zones_vent, self.agent.zones_thermiques, 
                        self.agent.zones_descendantes, self.agent.zones_inertie]
            for feature_list in features:
                feature_list[:] = [c for c in feature_list if c[0] < new_xy and c[1] < new_xy and c[2] < new_alt]
            
            # Reset Start/Target
            if self.agent.coords_depart:
                r, c, a = self.agent.coords_depart
                if r >= new_xy or c >= new_xy or a >= new_alt: self.agent.coords_depart = (new_xy-1, 0, 0)
            if self.agent.coords_cible:
                r, c, a = self.agent.coords_cible
                if r >= new_xy or c >= new_xy or a >= new_alt: self.agent.coords_cible = (0, new_xy-1, new_alt-1)
            
            # Update Visuals
            if new_xy > 15: self.CELL_SIZE = 40
            else: self.CELL_SIZE = 55
            self.canvas_width = self.CELL_SIZE * new_xy
            self.canvas_height = self.CELL_SIZE * new_xy
            self.canvas_grid.config(width=self.canvas_width, height=self.canvas_height)
            self.scale_alt.config(to=new_alt-1)
            self.current_altitude_level.set(0)
            self.draw_grid()
            messagebox.showinfo("Système", f"Géométrie mise à jour : {new_xy}x{new_xy} | {new_alt} Couches")
        else:
            messagebox.showinfo("Système", "Paramètres algorithmiques mis à jour.")

    def export_map(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not filepath: return
        data = {
            "taille_xy": self.agent.TAILLE_GRILLE_XY,
            "niveaux_altitude": self.agent.NIVEAUX_ALTITUDE,
            "depart": self.agent.coords_depart,
            "cible": self.agent.coords_cible,
            "obstacles": self.agent.coords_obstacles,
            "vent": self.agent.zones_vent,
            "thermiques": self.agent.zones_thermiques,
            "descendants": self.agent.zones_descendantes,
            "inertie": self.agent.zones_inertie
        }
        try:
            with open(filepath, 'w') as f: json.dump(data, f, indent=4)
            messagebox.showinfo("Export", "Sauvegarde réussie.")
        except Exception as e: messagebox.showerror("Erreur", str(e))

    def import_map(self):
        filepath = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not filepath: return
        try:
            with open(filepath, 'r') as f: data = json.load(f)
            self.agent.NIVEAUX_ALTITUDE = data.get("niveaux_altitude", 3)
            self.var_altitude_max.set(self.agent.NIVEAUX_ALTITUDE)
            self.scale_alt.config(to=self.agent.NIVEAUX_ALTITUDE-1)
            
            # Chargement coords
            self.agent.coords_depart = tuple(data["depart"]) if data["depart"] else None
            self.agent.coords_cible = tuple(data["cible"]) if data["cible"] else None
            self.agent.coords_obstacles = [tuple(x) for x in data["obstacles"]]
            self.agent.zones_vent = [tuple(x) for x in data["vent"]]
            self.agent.zones_thermiques = [tuple(x) for x in data["thermiques"]]
            self.agent.zones_descendantes = [tuple(x) for x in data["descendants"]]
            self.agent.zones_inertie = [tuple(x) for x in data["inertie"]]
            
            self.draw_grid()
            messagebox.showinfo("Import", "Configuration chargée.")
        except Exception as e: messagebox.showerror("Erreur", str(e))

    def show_help(self):
        help_win = tk.Toplevel(self.master)
        help_win.title("Manuel Technique")
        help_win.geometry("800x600")
        help_win.configure(bg="white")
        txt = tk.Text(help_win, wrap="word", font=("Segoe UI", 10), padx=40, pady=30, borderwidth=0)
        txt.pack(fill="both", expand=True)
        content = """
        NAVDRONE AI - SUITE DE SIMULATION
        =================================
        
        Ce simulateur utilise un algorithme Q-Learning (Reinforcement Learning) pour résoudre 
        des problèmes de navigation 3D complexes avec contraintes aérologiques.

        1. GUIDE RAPIDE
        ---------------
        - Utilisez la souris (Clic Gauche) sur la grille pour placer des éléments.
        - Changez d'altitude avec le curseur latéral pour éditer les différentes couches Z.
        - Placez obligatoirement un Départ (🛫) et une Cible (🎯).
        - Cliquez sur "LANCER SIMULATION" pour démarrer l'apprentissage.

        2. LÉGENDE TACTIQUE
        -------------------
        ⬛ Structure (Mur) : Obstacle infranchissable.
        〰 Turbulence : Zone de traînée augmentée (Coût énergétique).
        🔥 Ascendance : Courant thermique (Gain d'altitude favorable).
        ❄️ Rabattant : Courant descendant (Perte d'altitude forcée).
        ⚡ Cisaillement : Zone d'instabilité vectorielle (Dérive aléatoire).

        3. INTERPRÉTATION DES RÉSULTATS
        -------------------------------
        La courbe de convergence montre l'évolution du score moyen. Une courbe ascendante 
        indique que l'IA apprend correctement la topologie de la mission.
        """
        txt.insert("1.0", content)
        txt.config(state="disabled")

if __name__ == "__main__":
    DEFAULT_ALTITUDE = 3
    params = QLearningParameters(TAILLE_GRILLE_XY * TAILLE_GRILLE_XY * DEFAULT_ALTITUDE, NB_ACTIONS)
    agent = QLearningAgent(params, TAILLE_GRILLE_XY, DEFAULT_ALTITUDE)
    
    root = tk.Tk()
    app = QLearningGUI(root, agent, params)
    root.mainloop()
