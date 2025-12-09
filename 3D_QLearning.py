import numpy as np
import random
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import json 
from mpl_toolkits.mplot3d import Axes3D

# --- 1. CONFIGURATION & ESTHÉTIQUE ---
TAILLE_GRILLE_XY = 10 # Valeur par défaut, modifiable via l'UI

# Palette de couleurs "Aero Science"
THEME = {
    "bg_main": "#ECEFF1",    # Gris très clair (fond général)
    "bg_sidebar": "#263238", # Gris anthracite (panneau de contrôle)
    "text_sidebar": "#ECEFF1",
    "accent": "#039BE5",     # Bleu Aviation
    "accent_hover": "#0288D1",
    "success": "#43A047",    # Vert Indicateur
    "warning": "#FB8C00",    # Orange Alerte
    "danger": "#E53935",     # Rouge Erreur
    "card": "#FFFFFF"
}

# Couleurs des éléments physiques
COLORS = {
    "obstacle": "#37474F",   # Gris Sombre (Structure)
    "vent": "#90A4AE",       # Gris Bleu (Flux perturbé)
    "thermique": "#FF7043",  # Orange (Flux Ascendant)
    "descendant": "#29B6F6", # Cyan (Flux Descendant)
    "inertie": "#BA68C8",    # Violet (Cisaillement)
    "depart": "#66BB6A",     # Vert (Base)
    "cible": "#FFCA28",      # Jaune/Or (Objectif)
    "vide": "#FFFFFF",       # Blanc
    "grid": "#CFD8DC"        # Gris ligne
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

# --- 2. Outils UX (Tooltips & Helpers) ---
class ToolTip(object):
    """ Gestionnaire de bulles d'aide contextuelles """
    def __init__(self, widget, text='Info'):
        self.waittime = 500     #ms
        self.wraplength = 350   #pixels
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
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                       background="#37474F", foreground="#ECEFF1", relief='solid', borderwidth=0,
                       wraplength=self.wraplength, font=("Segoe UI", 9), padx=10, pady=6)
        label.pack()

    def hidetip(self):
        tw = self.tw
        self.tw= None
        if tw:
            tw.destroy()

# --- 3. Moteur de Simulation (Q-Learning) ---
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
    if not (0 <= r < taille_xy and 0 <= c < taille_xy and 0 <= a < niveaux_alt):
        return False
    if (r, c, a) in obstacles:
        return False
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
            # Dérive induite
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

# --- 4. Interface Graphique (GUI Enterprise) ---
class QLearningGUI:
    def __init__(self, master, agent, params):
        self.master = master
        self.agent = agent
        self.params = params
        
        master.title("NavDrone AI [Enterprise Edition] - Environnement de Simulation Avancé")
        master.geometry("1280x850")
        master.configure(bg=THEME["bg_main"])
        
        # Style Configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Styles Généraux
        self.style.configure("TFrame", background=THEME["bg_main"])
        self.style.configure("TLabel", background=THEME["bg_main"], foreground="#37474F", font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground=THEME["bg_sidebar"])
        self.style.configure("Card.TFrame", background=THEME["card"], relief="raised", borderwidth=1)
        
        # Style Sidebar (Sombre - Cockpit)
        self.style.configure("Sidebar.TFrame", background=THEME["bg_sidebar"])
        self.style.configure("Sidebar.TLabel", background=THEME["bg_sidebar"], foreground=THEME["text_sidebar"], font=("Segoe UI", 9))
        self.style.configure("SidebarTitle.TLabel", background=THEME["bg_sidebar"], foreground=THEME["accent"], font=("Segoe UI", 11, "bold"))
        self.style.configure("Sidebar.TLabelframe", background=THEME["bg_sidebar"], foreground=THEME["text_sidebar"])
        self.style.configure("Sidebar.TLabelframe.Label", background=THEME["bg_sidebar"], foreground=THEME["accent"], font=("Segoe UI", 10, "bold"))
        
        # Boutons
        self.style.configure("Action.TButton", font=("Segoe UI", 10, "bold"), background=THEME["accent"], foreground="white", padding=10)
        self.style.map("Action.TButton", background=[('active', THEME["accent_hover"])])
        
        self.style.configure("Success.TButton", font=("Segoe UI", 11, "bold"), background=THEME["success"], foreground="white", padding=10)
        self.style.map("Success.TButton", background=[('active', '#2E7D32')])

        self.style.configure("TNotebook", background=THEME["bg_main"], tabposition='n')
        self.style.configure("TNotebook.Tab", padding=[20, 8], font=("Segoe UI", 11, "bold"), background="#CFD8DC")
        self.style.map("TNotebook.Tab", background=[("selected", "white")], foreground=[("selected", THEME["accent"])])

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
        
        self.setup_ui()
        
    def setup_ui(self):
        # Header
        header = ttk.Frame(self.master, padding=15)
        header.pack(fill='x')
        
        lbl_title = ttk.Label(header, text="🚁 NavDrone AI Control Center", style="Header.TLabel")
        lbl_title.pack(side=tk.LEFT, padx=10)
        
        btn_help = ttk.Button(header, text="Documentation Technique", command=self.show_help)
        btn_help.pack(side=tk.RIGHT)
        ToolTip(btn_help, "Consulter les spécifications techniques et le manuel d'opération.")

        # Contenu principal
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(pady=5, padx=10, fill="both", expand=True)

        self.frame_editor = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_editor, text="   DESIGN & SIMULATION   ")
        self.setup_editor_tab(self.frame_editor)

        self.frame_3d = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_3d, text="   VISUALISATION 3D   ")
        self.setup_3d_frame(self.frame_3d)

        self.frame_stats = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_stats, text="   TÉLÉMETRIE   ")
        self.setup_stats_tab(self.frame_stats)
        
        self.frame_params = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_params, text="   PARAMÈTRES SYSTÈME   ")
        self.setup_params_tab(self.frame_params)

        self.draw_grid()
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

    # --- TAB 1: ÉDITEUR (UX Pro) ---
    def setup_editor_tab(self, parent):
        # 1. ZONE CARTE (Gauche)
        frame_map_container = ttk.Frame(parent, padding=15)
        frame_map_container.pack(side=tk.LEFT, fill="both", expand=True)
        
        map_header = ttk.Frame(frame_map_container)
        map_header.pack(fill='x', pady=(0,10))
        self.lbl_etage = ttk.Label(map_header, text="ALTITUDE (AGL) : 0", font=("Segoe UI", 14, "bold"), foreground=THEME["accent"])
        self.lbl_etage.pack(side=tk.LEFT)
        
        canvas_frame = tk.Frame(frame_map_container, bg="gray", bd=1) 
        canvas_frame.pack(anchor="center")
        
        self.canvas_grid = tk.Canvas(canvas_frame, width=self.canvas_width, height=self.canvas_height, 
                                     bg="white", highlightthickness=0)
        self.canvas_grid.pack()
        self.canvas_grid.bind("<Button-1>", self.on_grid_click)
        ToolTip(self.canvas_grid, "Zone Tactique.\nClic gauche pour placer les entités sélectionnées.\nLes éléments dépendent de l'altitude Z actuelle.")

        # 2. ZONE COMMANDES (Droite)
        sidebar = ttk.Frame(parent, width=350, style="Sidebar.TFrame")
        sidebar.pack(side=tk.RIGHT, fill="y", padx=0, pady=0, ipadx=10)
        
        ttk.Label(sidebar, text="CONTRÔLE DE MISSION", style="SidebarTitle.TLabel", padding=15).pack(fill='x')

        # Altitude
        lf_alt = ttk.LabelFrame(sidebar, text="LAYER CONTROL (ALTITUDE)", padding=10, style="Sidebar.TLabelframe")
        lf_alt.pack(fill='x', pady=5, padx=10)
        
        self.scale_alt = tk.Scale(lf_alt, from_=0, to=self.agent.NIVEAUX_ALTITUDE-1, orient=tk.HORIZONTAL, 
                                   variable=self.current_altitude_level, command=self.on_altitude_change,
                                   bg=THEME["bg_sidebar"], fg="white", highlightthickness=0, troughcolor="#37474F", activebackground=THEME["accent"])
        self.scale_alt.pack(fill='x', pady=5)
        ttk.Label(lf_alt, text="Sélecteur de Couche Z", style="Sidebar.TLabel", font=("Segoe UI", 8)).pack(anchor='e')

        # Outils
        lf_tools = ttk.LabelFrame(sidebar, text="OUTILS ENVIRONNEMENT", padding=10, style="Sidebar.TLabelframe")
        lf_tools.pack(fill='x', pady=10, padx=10)
        
        # Icônes vectorielles propres et claires
        tools = [
            ("Départ Drone", "depart", COLORS["depart"], "🛫"),   
            ("Cible", "cible", COLORS["cible"], "🎯"),
            ("Structure", "obstacle", COLORS["obstacle"], "⬛"),
            ("Effacer", "effacer", "#B0BEC5", "🧹"),
            
            # Phénomènes Aérologiques
            ("Turbulences", "vent", COLORS["vent"], "〰"),         # Vague pour l'air
            ("Ascendance", "thermique", COLORS["thermique"], "⇧"), # Flèche haut
            ("Rabattant", "descendant", COLORS["descendant"], "⇩"),# Flèche bas
            ("Cisaillement", "inertie", COLORS["inertie"], "⚡")   # Éclair
        ]
        
        frame_palette = tk.Frame(lf_tools, bg=THEME["bg_sidebar"])
        frame_palette.pack(fill='x')
        
        for i, (name, mode, col, icon) in enumerate(tools):
            row = i // 2
            col_idx = i % 2
            btn_container = tk.Frame(frame_palette, bg=THEME["bg_sidebar"], pady=3, padx=3)
            btn_container.grid(row=row, column=col_idx, sticky="ew")
            
            rb = tk.Radiobutton(btn_container, text=f"{icon} {name}", variable=self.edit_mode, value=mode,
                                indicatoron=0, width=16, 
                                bg="#37474F", fg="white", selectcolor=THEME["accent"], 
                                activebackground=THEME["accent_hover"], activeforeground="white",
                                font=("Segoe UI", 9), relief="flat", bd=0, pady=8)
            rb.pack(fill="both")
            
            # Descriptions techniques
            desc = {
                "depart": "Point d'insertion initial [État S0].\nLe drone démarre ici.",
                "cible": "Objectif de mission [État Terminal].\nRécompense maximale (+1000).",
                "obstacle": "Zone d'exclusion statique (NFZ).\nCollision entraîne un échec immédiat.",
                "vent": "Zone de turbulence atmosphérique.\nAugmente le coût de traversée (Pénalité énergétique).",
                "thermique": "Courant ascendant thermique.\nFavorise le mouvement vertical positif (Gain Z).",
                "descendant": "Flux d'air rabattant.\nForce vectorielle verticale négative (Perte Z).",
                "inertie": "Zone de cisaillement (Wind Shear).\nInduit une dérive stochastique imprévisible.",
                "effacer": "Suppression des entités sur la coordonnée active."
            }
            ToolTip(rb, desc.get(mode, ""))
        
        frame_palette.columnconfigure(0, weight=1)
        frame_palette.columnconfigure(1, weight=1)

        # Opérations
        lf_ops = ttk.Frame(sidebar, style="Sidebar.TFrame")
        lf_ops.pack(fill='x', pady=20, padx=10)
        
        self.btn_train = ttk.Button(lf_ops, text="▶  INITIALISER SIMULATION", style="Success.TButton", command=self.start_training_manually)
        self.btn_train.pack(fill='x', pady=5)
        
        self.progress = ttk.Progressbar(lf_ops, length=100, mode='determinate')
        self.progress.pack(fill='x', pady=2)
        
        self.lbl_status = ttk.Label(lf_ops, text="Système Prêt.", style="Sidebar.TLabel", font=("Segoe UI", 9, "italic"))
        self.lbl_status.pack(anchor='w', pady=(2, 10))

        self.btn_replay = ttk.Button(lf_ops, text="↺  VISUALISER TRAJECTOIRE", style="Action.TButton", command=self.start_replay, state=tk.DISABLED)
        self.btn_replay.pack(fill='x', pady=5)

        # Data
        frame_data = tk.Frame(sidebar, bg=THEME["bg_sidebar"])
        frame_data.pack(side=tk.BOTTOM, fill='x', padx=10, pady=10)
        
        btn_save = tk.Button(frame_data, text="💾 Export JSON", bg="#546E7A", fg="white", relief="flat", command=self.export_map)
        btn_save.pack(side=tk.LEFT, fill='x', expand=True, padx=(0,2))
        
        btn_load = tk.Button(frame_data, text="📂 Import JSON", bg="#546E7A", fg="white", relief="flat", command=self.import_map)
        btn_load.pack(side=tk.RIGHT, fill='x', expand=True, padx=(2,0))

    # --- TAB 2: 3D ---
    def setup_3d_frame(self, frame):
        plot_container = ttk.Frame(frame, style="Card.TFrame", padding=15)
        plot_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        self.fig_3d = plt.figure(figsize=(5, 4), facecolor='white')
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=plot_container)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        lbl = ttk.Label(plot_container, text="Modèle 3D - Rotation via clic gauche + glisser", background="white", foreground="gray")
        lbl.pack(pady=5)
        self.draw_3d_environment([])

    # --- TAB 3: STATS ---
    def setup_stats_tab(self, frame):
        paned = ttk.PanedWindow(frame, orient=tk.HORIZONTAL)
        paned.pack(fill='both', expand=True, padx=20, pady=20)
        
        f1 = ttk.Frame(paned, style="Card.TFrame", padding=15)
        paned.add(f1, weight=1)
        ttk.Label(f1, text="Analyse de Convergence (Reward/Episode)", style="Header.TLabel", background="white").pack(anchor='w', pady=(0,10))
        
        self.fig_conv, self.ax_conv = plt.subplots(figsize=(4,3), facecolor='white')
        self.canvas_chart = FigureCanvasTkAgg(self.fig_conv, master=f1)
        self.canvas_chart.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        f2 = ttk.Frame(paned, style="Card.TFrame", padding=15)
        paned.add(f2, weight=1)
        ttk.Label(f2, text="Profil de Vol (Altitude & Score Cumulé)", style="Header.TLabel", background="white").pack(anchor='w', pady=(0,10))
        
        self.fig_stats, (self.ax_alt, self.ax_reward) = plt.subplots(2, 1, figsize=(4, 4), sharex=True, facecolor='white')
        self.canvas_stats = FigureCanvasTkAgg(self.fig_stats, master=f2)
        self.canvas_stats.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- TAB 4: PARAMÈTRES ---
    def setup_params_tab(self, frame):
        container = ttk.Frame(frame, style="Card.TFrame", padding=30)
        container.pack(fill='both', expand=True, padx=100, pady=30)
        
        ttk.Label(container, text="Configuration du Solveur Q-Learning", style="Header.TLabel", background="white").pack(anchor='w', pady=(0, 20))
        
        grid_f = ttk.Frame(container, style="Card.TFrame")
        grid_f.pack(fill='x')
        
        def create_param_row(parent, row, label_text, var, tooltip_text, is_int=False):
            ttk.Label(parent, text=label_text, background="white", font=("Segoe UI", 11, "bold")).grid(row=row, column=0, sticky='w', pady=12, padx=10)
            entry = ttk.Entry(parent, textvariable=var, width=12, font=("Segoe UI", 11))
            entry.grid(row=row, column=1, sticky='w', pady=12)
            ToolTip(entry, tooltip_text)
            unit = "(int)" if is_int else "(float)"
            ttk.Label(parent, text=unit, background="white", foreground="gray").grid(row=row, column=2, sticky='w', padx=5)

        self.var_episodes = tk.IntVar(value=self.params.EPISODES)
        self.var_alpha = tk.DoubleVar(value=self.params.ALPHA)
        self.var_gamma = tk.DoubleVar(value=self.params.GAMMA)
        self.var_epsilon = tk.DoubleVar(value=self.params.EPSILON_INIT)
        
        # Dimensions Map
        self.var_altitude_max = tk.IntVar(value=self.agent.NIVEAUX_ALTITUDE)
        self.var_taille_xy = tk.IntVar(value=self.agent.TAILLE_GRILLE_XY)

        create_param_row(grid_f, 0, "Itérations (Épisodes) :", self.var_episodes, "Nombre total de cycles d'entraînement pour la convergence.", True)
        create_param_row(grid_f, 1, "Taux d'Apprentissage (α) :", self.var_alpha, "Poids donné aux nouvelles informations vs connaissances acquises.")
        create_param_row(grid_f, 2, "Facteur d'Actualisation (γ) :", self.var_gamma, "Importance des récompenses futures (0=Myope, 1=Visionnaire).")
        create_param_row(grid_f, 3, "Exploration Initiale (ε) :", self.var_epsilon, "Probabilité de choix d'action aléatoire au démarrage.")
        
        ttk.Separator(grid_f, orient='horizontal').grid(row=4, column=0, columnspan=3, sticky='ew', pady=20)
        
        create_param_row(grid_f, 5, "Dimensions Grille (X/Y) :", self.var_taille_xy, "Taille latérale de la zone de vol (Carré). Min: 5, Max: 20.", True)
        create_param_row(grid_f, 6, "Plafond de Vol (Couches Z) :", self.var_altitude_max, "Hauteur maximale de l'espace aérien discrétisé. Min: 1, Max: 10.", True)

        btn_apply = ttk.Button(container, text="Appliquer Configuration", style="Action.TButton", command=self.apply_params)
        btn_apply.pack(pady=30)
        
        self.lbl_advice = ttk.Label(container, text="", foreground=THEME["accent"], background="white", font=("Segoe UI", 10, "italic"))
        self.lbl_advice.pack()
        self.update_advice()

    # --- LOGIQUE DESSIN & ETATS ---
    def on_tab_change(self, event):
        tab_id = self.notebook.index(self.notebook.select())
        if tab_id == 1: 
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
        self.update_advice()

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
        
        for i in range(self.agent.TAILLE_GRILLE_XY + 1):
            self.canvas_grid.create_line(0, i*sz, self.canvas_width, i*sz, fill=COLORS["grid"])
            self.canvas_grid.create_line(i*sz, 0, i*sz, self.canvas_height, fill=COLORS["grid"])

        def draw_cell(r, c, color, text="", text_color="white"):
            x, y = c*sz, r*sz
            self.canvas_grid.create_rectangle(x+1, y+1, x+sz-1, y+sz-1, fill=color, outline=color)
            if text:
                # Adaptation taille police si la case est petite
                font_size = 16 if sz >= 50 else 10
                self.canvas_grid.create_text(x+sz/2, y+sz/2, text=text, fill=text_color, font=('Segoe UI', font_size, 'bold'))

        for r in range(self.agent.TAILLE_GRILLE_XY):
            for c in range(self.agent.TAILLE_GRILLE_XY):
                coord = (r, c, view_a)
                
                if coord in self.agent.coords_obstacles: draw_cell(r, c, COLORS["obstacle"])
                elif coord in self.agent.zones_vent: draw_cell(r, c, COLORS["vent"], "〰")
                elif coord in self.agent.zones_thermiques: draw_cell(r, c, COLORS["thermique"], "⇧")
                elif coord in self.agent.zones_descendantes: draw_cell(r, c, COLORS["descendant"], "⇩")
                elif coord in self.agent.zones_inertie: draw_cell(r, c, COLORS["inertie"], "⚡")
                
                if coord == self.agent.coords_depart: draw_cell(r, c, COLORS["depart"], "🛫")
                if coord == self.agent.coords_cible: draw_cell(r, c, COLORS["cible"], "🎯")

        if path:
            for i, (r, c, a) in enumerate(path):
                if a == view_a:
                    cx, cy = c*sz+sz/2, r*sz+sz/2
                    col = "#FFD600" if i==len(path)-1 else THEME["accent"]
                    self.canvas_grid.create_oval(cx-6, cy-6, cx+6, cy+6, fill=col, outline="white", width=2)
                    
                    if i>0:
                         pr, pc, pa = path[i-1]
                         if pa == a:
                             self.canvas_grid.create_line(pc*sz+sz/2, pr*sz+sz/2, cx, cy, fill=THEME["accent"], width=3, capstyle=tk.ROUND)
                         else:
                             txt = "▲" if pa < a else "▼"
                             offset_y = -15 if pa < a else 15
                             self.canvas_grid.create_text(cx+15, cy+offset_y, text=txt, fill="#D500F9", font=('Arial', 16, 'bold'))

    # --- 3D RENDERING ---
    def draw_3d_environment(self, path=None):
        if not self.master.winfo_exists(): return
        self.ax_3d.clear()
        self.ax_3d.view_init(elev=35, azim=-50)
        
        max_c = self.agent.TAILLE_GRILLE_XY - 1
        max_a = self.agent.NIVEAUX_ALTITUDE - 1
        self.ax_3d.set_xlim(0, max_c+1); self.ax_3d.set_ylim(0, max_c+1); self.ax_3d.set_zlim(0, max_a+1)
        self.ax_3d.set_xlabel("X (Est)"); self.ax_3d.set_ylabel("Y (Nord)"); self.ax_3d.set_zlabel("Z (Alt)")
        self.ax_3d.invert_yaxis()

        def draw_voxels(coord_list, color, alpha):
            for r, c, a in coord_list:
                self.ax_3d.bar3d(c, r, a, 1, 1, 1, color=color, alpha=alpha, shade=True, edgecolor='gray', linewidth=0.1)

        draw_voxels(self.agent.coords_obstacles, COLORS["obstacle"], 0.9)
        draw_voxels(self.agent.zones_thermiques, COLORS["thermique"], 0.4)
        draw_voxels(self.agent.zones_descendantes, COLORS["descendant"], 0.4)
        draw_voxels(self.agent.zones_vent, COLORS["vent"], 0.3)
        draw_voxels(self.agent.zones_inertie, COLORS["inertie"], 0.3)

        if self.agent.coords_depart:
            d = self.agent.coords_depart
            self.ax_3d.scatter(d[1]+0.5, d[0]+0.5, d[2]+0.5, c=COLORS["depart"], s=150, label="Départ", edgecolors='white')
        if self.agent.coords_cible:
            t = self.agent.coords_cible
            self.ax_3d.scatter(t[1]+0.5, t[0]+0.5, t[2]+0.5, c=COLORS["cible"], marker='*', s=300, label="Cible", edgecolors='black')

        if path:
            xs = [p[1]+0.5 for p in path]
            ys = [p[0]+0.5 for p in path]
            zs = [p[2]+0.5 for p in path]
            self.ax_3d.plot(xs, ys, zs, c="#2962FF", linewidth=3, marker='o', markersize=4, label="Trajectoire")
        
        self.canvas_3d.draw()

    # --- ENTRAÎNEMENT ---
    def start_training_manually(self):
        if not self.agent.coords_depart or not self.agent.coords_cible:
            messagebox.showwarning("Erreur Configuration", "Définition incomplète : Point de Départ (🛫) et Cible (🎯) requis.")
            return
        
        self.training_in_progress = True
        self.btn_train.config(state=tk.DISABLED)
        self.btn_replay.config(state=tk.DISABLED)
        self.lbl_status['text'] = "Calcul de trajectoire en cours..."
        self.lbl_status.config(foreground=THEME["warning"])
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
        self.lbl_status['text'] = f"Simulation : {ep}/{self.params.EPISODES} | Reward Max : {rew:.1f}"

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
        self.lbl_status['text'] = "Convergence Atteinte. Solution Disponible."
        self.lbl_status.config(foreground=THEME["success"])
        self.progress['value'] = 100
        
        self.final_optimal_path = obtenir_chemin_optimal(
            self.agent.Q_table, self.agent.coords_depart, self.agent.coords_cible,
            self.agent.coords_obstacles, self.agent.zones_vent,
            self.agent.zones_thermiques, self.agent.zones_descendantes, self.agent.zones_inertie,
            self.agent.TAILLE_GRILLE_XY, self.agent.NIVEAUX_ALTITUDE
        )
        
        self.draw_grid(self.final_optimal_path)
        self.plot_graphs()
        messagebox.showinfo("Fin de Simulation", "Le modèle a convergé.\nAnalysez la solution via le Replay ou la vue 3D.")

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
            self.master.after(200, self.loop_replay)
        else:
            self.replay_running = False
            self.btn_replay.config(state=tk.NORMAL)

    def plot_graphs(self):
        self.ax_conv.clear()
        if self.agent.historique_recompenses:
            self.ax_conv.plot(self.agent.historique_recompenses, color='#B0BEC5', alpha=0.5, label="Signal Brut")
            w = 50
            if len(self.agent.historique_recompenses) > w:
                avg = np.convolve(self.agent.historique_recompenses, np.ones(w)/w, mode='valid')
                self.ax_conv.plot(np.arange(w-1, len(self.agent.historique_recompenses)), avg, color=THEME["accent"], label="Moyenne Mobile", linewidth=2)
        
        self.ax_conv.set_title("Stabilité de l'Apprentissage", fontsize=10)
        self.ax_conv.grid(True, linestyle=':', alpha=0.6)
        self.canvas_chart.draw()
        
        p = self.final_optimal_path
        self.ax_alt.clear(); self.ax_reward.clear()
        if p and len(p) > 1:
            steps = range(len(p))
            self.ax_alt.step(steps, [x[2] for x in p], color=THEME["accent"], where='mid', linewidth=2)
            self.ax_alt.set_ylabel("Altitude Z")
            self.ax_alt.grid(True, alpha=0.3)
            
            rewards = []
            cum = 0
            for i in range(len(p)-1):
                curr, nxt = p[i], p[i+1]
                r_step = -1 
                if nxt == self.agent.coords_cible: r_step += 1000
                cum += r_step
                rewards.append(cum)
            
            self.ax_reward.plot(rewards, color=THEME["success"], linewidth=2)
            self.ax_reward.set_ylabel("Score Cumulé")
            self.ax_reward.set_xlabel("Waypoints")
            self.ax_reward.grid(True, alpha=0.3)

        self.fig_stats.tight_layout()
        self.canvas_stats.draw()

    # --- PARAMÈTRES & DOC ---
    def apply_params(self):
        if self.training_in_progress: 
            messagebox.showwarning("Opération Impossible", "Simulation en cours d'exécution.")
            return

        self.params.EPISODES = self.var_episodes.get()
        self.params.ALPHA = self.var_alpha.get()
        self.params.GAMMA = self.var_gamma.get()
        self.params.EPSILON_INIT = self.var_epsilon.get()

        # Nouvelle logique de redimensionnement
        new_xy = self.var_taille_xy.get()
        new_xy = max(5, min(20, new_xy)) # Sécurité pour l'UI

        new_alt = self.var_altitude_max.get()
        new_alt = max(1, min(10, new_alt)) 
        
        if new_xy != self.agent.TAILLE_GRILLE_XY or new_alt != self.agent.NIVEAUX_ALTITUDE:
            # Update Agent
            self.agent.TAILLE_GRILLE_XY = new_xy
            self.agent.NIVEAUX_ALTITUDE = new_alt
            self.agent.TAILLE_ETAT = new_xy * new_xy * new_alt
            self.agent.reset_for_training()
            
            # Nettoyage des objets hors limites (XY et Z)
            features = [self.agent.coords_obstacles, self.agent.zones_vent, self.agent.zones_thermiques, 
                        self.agent.zones_descendantes, self.agent.zones_inertie]
            for feature_list in features:
                feature_list[:] = [c for c in feature_list if c[0] < new_xy and c[1] < new_xy and c[2] < new_alt]
            
            # Reset Start/Target si hors limites
            if self.agent.coords_depart:
                r, c, a = self.agent.coords_depart
                if r >= new_xy or c >= new_xy or a >= new_alt:
                    self.agent.coords_depart = (new_xy-1, 0, 0)
            
            if self.agent.coords_cible:
                r, c, a = self.agent.coords_cible
                if r >= new_xy or c >= new_xy or a >= new_alt:
                    self.agent.coords_cible = (0, new_xy-1, new_alt-1)
            
            # Mise à jour de l'UI Canvas
            # Adaptation de la taille des cellules pour que ça rentre dans l'écran
            if new_xy > 15:
                self.CELL_SIZE = 40
            else:
                self.CELL_SIZE = 55
                
            self.canvas_width = self.CELL_SIZE * new_xy
            self.canvas_height = self.CELL_SIZE * new_xy
            self.canvas_grid.config(width=self.canvas_width, height=self.canvas_height)
            
            # Reset Sliders
            self.current_altitude_level.set(0)
            self.scale_alt.config(to=new_alt-1)
            
            self.draw_grid()
            messagebox.showinfo("Système", f"Géométrie redéfinie : {new_xy}x{new_xy} sur {new_alt} couches.")
        
        self.update_advice()
        messagebox.showinfo("Système", "Paramètres enregistrés.")

    def update_advice(self):
        c = self.params.calculer_episodes_conseilles(self.agent.coords_obstacles)
        self.lbl_advice.config(text=f"ℹ Estimation : Convergence attendue vers ~{c} itérations.")

    def show_help(self):
        help_win = tk.Toplevel(self.master)
        help_win.title("Manuel Technique de Simulation")
        help_win.geometry("800x650")
        help_win.configure(bg="white")
        
        txt = tk.Text(help_win, wrap="word", font=("Segoe UI", 10), padx=40, pady=30, borderwidth=0)
        txt.pack(fill="both", expand=True)
        
        content = """
        MANUEL TECHNIQUE DE SIMULATION & CONTRÔLE DE VOL
        ==================================================

        1. ARCHITECTURE DU SYSTÈME
        --------------------------
        Ce logiciel implémente un agent autonome basé sur l'apprentissage par renforcement (Q-Learning) 
        pour la navigation de drone en environnement 3D contraint. L'agent apprend une politique optimale π(s) 
        en maximisant la récompense cumulée à travers des épisodes d'exploration/exploitation.

        2. PARAMÉTRAGE DE L'ESPACE AÉRIEN
        ---------------------------------
        L'environnement est discrétisé en voxels (Cases 3D). L'utilisateur doit définir la topologie 
        avant de lancer la simulation.
        
        [Contrôles]
        > ALTITUDE (AGL) : Utilisez le slider latéral pour naviguer entre les couches Z (0=Sol, Max=Plafond).
        > Outils de Design : Sélectionnez un outil et cliquez sur la grille pour modifier l'état du voxel.

        [Légende des Entités]
        🛫 DÉPART (Start) : Point d'insertion du vecteur.
        🎯 CIBLE (Target) : Coordonnée objectif. Atteindre ce point termine l'épisode avec succès.
        ⬛ OBSTACLE (NFZ) : Structure solide. Collision fatale.

        3. DYNAMIQUE DES FLUIDES & CONTRAINTES
        --------------------------------------
        L'environnement simule des phénomènes aérologiques affectant la consommation énergétique 
        et la stabilité du vol.

        〰 TURBULENCES (Drag) : Zone de haute densité/friction. Augmente le coût de déplacement.
           Impact : Pénalité de score (-4).
        
        ⇧ ASCENDANCE THERMIQUE (Lift) : Flux d'air vertical chaud.
           Impact : Facilite l'ascension (Bonus +5). Pénalise la descente contre le flux.
        
        ⇩ FLUX RABATTANT (Sink) : Flux d'air vertical froid.
           Impact : Facilite la descente (Bonus +5). Pénalise l'ascension contre le flux.
        
        ⚡ CISAILLEMENT (Wind Shear) : Instabilité vectorielle majeure.
           Impact : Introduit une composante stochastique (50% de probabilité de dérive latérale involontaire).

        4. PROTOCOLE D'EXPÉRIMENTATION
        ------------------------------
        A. Initialisation : Définir Départ, Cible et Obstacles.
        B. Simulation : Lancer l'entraînement via "INITIALISER SIMULATION".
        C. Analyse : Vérifier la convergence via la courbe "Stabilité de l'Apprentissage".
        D. Validation : Visualiser la trajectoire finale via le Replay ou la Vue 3D.
        """
        txt.insert("1.0", content)
        txt.config(state="disabled")

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
            messagebox.showinfo("Export", "Cartographie exportée avec succès.")
        except Exception as e: messagebox.showerror("Erreur", str(e))

    def import_map(self):
        filepath = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not filepath: return
        try:
            with open(filepath, 'r') as f: data = json.load(f)
            if data["taille_xy"] != self.agent.TAILLE_GRILLE_XY:
                messagebox.showerror("Erreur", "Incompatibilité dimensionnelle.")
                return
            
            self.agent.NIVEAUX_ALTITUDE = data.get("niveaux_altitude", 3)
            self.scale_alt.config(to=self.agent.NIVEAUX_ALTITUDE-1)
            self.var_altitude_max.set(self.agent.NIVEAUX_ALTITUDE)
            
            self.agent.coords_depart = tuple(data["depart"]) if data["depart"] else None
            self.agent.coords_cible = tuple(data["cible"]) if data["cible"] else None
            self.agent.coords_obstacles = [tuple(x) for x in data["obstacles"]]
            self.agent.zones_vent = [tuple(x) for x in data["vent"]]
            self.agent.zones_thermiques = [tuple(x) for x in data["thermiques"]]
            self.agent.zones_descendantes = [tuple(x) for x in data["descendants"]]
            self.agent.zones_inertie = [tuple(x) for x in data["inertie"]]
            
            self.draw_grid()
            self.update_advice()
            messagebox.showinfo("Import", "Cartographie chargée.")
        except Exception as e: messagebox.showerror("Erreur", str(e))

if __name__ == "__main__":
    DEFAULT_ALTITUDE = 3
    params = QLearningParameters(TAILLE_GRILLE_XY * TAILLE_GRILLE_XY * DEFAULT_ALTITUDE, NB_ACTIONS)
    agent = QLearningAgent(params, TAILLE_GRILLE_XY, DEFAULT_ALTITUDE)
    
    root = tk.Tk()
    app = QLearningGUI(root, agent, params)
    root.mainloop()
