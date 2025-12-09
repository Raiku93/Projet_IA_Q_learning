import numpy as np
import random
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import threading
import json 
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Paramètres Globaux et Constantes ---
TAILLE_GRILLE_XY = 10
# Note : NIVEAUX_ALTITUDE est défini dynamiquement au lancement de l'application

ACTIONS = {
    0: 'HAUT_2D',    # (r-1, c, a)
    1: 'BAS_2D',     # (r+1, c, a)
    2: 'GAUCHE',     # (r, c-1, a)
    3: 'DROITE',     # (r, c+1, a)
    4: 'MONTER',     # (r, c, a+1)
    5: 'DESCENDRE'   # (r, c, a-1)
}
NB_ACTIONS = len(ACTIONS)

# --- Récompenses & Physique ---
RECOMPENSE_CIBLE = 1000
RECOMPENSE_COLLISION = -50
RECOMPENSE_MOUVEMENT_2D = -1
RECOMPENSE_MOUVEMENT_3D = -3
RECOMPENSE_PENALITE_VENT = -4
RECOMPENSE_BONUS_FLUX = 5       
RECOMPENSE_MALUS_CONTRE_FLUX = -10 
PROBABILITE_GLISSEMENT = 0.5    

# --- Couleurs pour l'UI et la 3D ---
COLORS = {
    "obstacle": "black",
    "vent": "lightblue",
    "thermique": "orange",
    "descendant": "cyan",
    "inertie": "yellow",
    "depart": "red",
    "cible": "green",
    "vide": "lightgray"
}

# --- 2. Classe de Paramètres ---
class QLearningParameters:
    def __init__(self, taille_etat_base, nb_actions):
        self.TAILLE_ETAT_BASE = taille_etat_base
        self.NB_ACTIONS = nb_actions
        # Hyperparamètres par défaut
        self.EPISODES = 10000 
        self.ALPHA = 0.1 
        self.GAMMA = 0.9 
        self.EPSILON_INIT = 1.0
        self.DECAY_EPSILON = 0.9996 
        self.MIN_EPSILON = 0.01

    def calculer_episodes_conseilles(self, coords_obstacles):
        """Calcule un nombre d'épisodes conseillé selon la complexité."""
        nombre_etats_libres = self.TAILLE_ETAT_BASE - len(coords_obstacles)
        facteur_complexite = 1.5 
        conseil = int(nombre_etats_libres * 20 * facteur_complexite)
        return max(10000, min(30000, conseil))

# --- 3. Fonctions Logiques de l'Environnement ---

def coords_to_idx(coords, taille_xy, niveaux_altitude):
    r, c, a = coords
    taille_couche_xy = taille_xy * taille_xy
    return a * taille_couche_xy + r * taille_xy + c

def initialiser_positions_vide(taille_xy, niveaux_alt):
    """Initialise une map vide, seulement Départ et Cible."""
    coords_depart = (taille_xy - 1, 0, 0) # Coin bas gauche, sol
    coords_cible = (0, taille_xy - 1, niveaux_alt - 1) # Coin haut droite, plafond
    
    # Listes vides au démarrage
    coords_obstacles = []
    zones_vent = []
    zones_thermiques = []    
    zones_descendantes = []  
    zones_inertie = []       
    
    return coords_depart, coords_cible, coords_obstacles, zones_vent, zones_thermiques, zones_descendantes, zones_inertie

def deplacer_simple(r, c, a, action_index):
    """Calcule la coordonnée brute après mouvement."""
    nr, nc, na = r, c, a
    if action_index == 0: nr -= 1    
    elif action_index == 1: nr += 1  
    elif action_index == 2: nc -= 1  
    elif action_index == 3: nc += 1  
    elif action_index == 4: na += 1  
    elif action_index == 5: na -= 1  
    return nr, nc, na

def est_valide(r, c, a, taille_xy, niveaux_alt, obstacles):
    """Vérifie les limites de la grille et les obstacles."""
    if not (0 <= r < taille_xy and 0 <= c < taille_xy and 0 <= a < niveaux_alt):
        return False
    if (r, c, a) in obstacles:
        return False
    return True

def obtenir_etat_recompense_suivants(etat_courant, action_index, coords_cible, coords_obstacles, 
                                     zones_vent, zones_thermiques, zones_descendantes, zones_inertie,
                                     taille_xy, niveaux_alt, mode_stochastique=True):
    """Cœur du moteur physique et des récompenses."""
    r, c, a = etat_courant
    new_r, new_c, new_a = deplacer_simple(r, c, a, action_index)
    
    # Collision Mur ou Obstacle immédiat
    if not est_valide(new_r, new_c, new_a, taille_xy, niveaux_alt, coords_obstacles):
        return etat_courant, RECOMPENSE_COLLISION, True 

    recompense_mouvement = 0
    etat_intermediaire = (new_r, new_c, new_a)

    # Coût de base du mouvement
    if action_index in [0, 1, 2, 3]: recompense_mouvement = RECOMPENSE_MOUVEMENT_2D
    elif action_index in [4, 5]: recompense_mouvement = RECOMPENSE_MOUVEMENT_3D

    # 1. Vent (Pénalité de zone)
    if etat_intermediaire in zones_vent:
        recompense_mouvement += RECOMPENSE_PENALITE_VENT

    # 2. Thermique (Bonus si on monte, Malus si on descend)
    if etat_courant in zones_thermiques:
        if action_index == 4: recompense_mouvement += RECOMPENSE_BONUS_FLUX 
        elif action_index == 5: recompense_mouvement += RECOMPENSE_MALUS_CONTRE_FLUX 
            
    # 3. Descendant (Bonus si on descend, Malus si on monte)
    if etat_courant in zones_descendantes:
        if action_index == 5: recompense_mouvement += RECOMPENSE_BONUS_FLUX
        elif action_index == 4: recompense_mouvement += RECOMPENSE_MALUS_CONTRE_FLUX

    etat_final = etat_intermediaire
    
    # 4. Inertie (Glissement Stochastique)
    if etat_intermediaire in zones_inertie and action_index in [0, 1, 2, 3]:
        # Si on bouge horizontalement dans une zone d'inertie
        if mode_stochastique and random.random() < PROBABILITE_GLISSEMENT:
            # On glisse d'une case de plus dans la même direction
            slide_r, slide_c, slide_a = deplacer_simple(new_r, new_c, new_a, action_index)
            
            # Si le glissement tape un mur
            if not est_valide(slide_r, slide_c, slide_a, taille_xy, niveaux_alt, coords_obstacles):
                return etat_intermediaire, RECOMPENSE_COLLISION, True
            else:
                etat_final = (slide_r, slide_c, slide_a)

    # Vérification Cible
    if etat_final == coords_cible:
        return etat_final, RECOMPENSE_CIBLE + recompense_mouvement, True
    
    return etat_final, recompense_mouvement, False

# --- 4. Agent Q-Learning ---
class QLearningAgent:
    def __init__(self, params: QLearningParameters, taille_xy, niveaux_altitude):
        self.params = params
        self.TAILLE_GRILLE_XY = taille_xy
        self.NIVEAUX_ALTITUDE = niveaux_altitude
        
        # Initialisation avec MAP VIDE
        self.coords_depart, self.coords_cible, self.coords_obstacles, self.zones_vent, \
        self.zones_thermiques, self.zones_descendantes, self.zones_inertie = \
            initialiser_positions_vide(self.TAILLE_GRILLE_XY, self.NIVEAUX_ALTITUDE)
            
        self.TAILLE_ETAT = self.TAILLE_GRILLE_XY * self.TAILLE_GRILLE_XY * self.NIVEAUX_ALTITUDE
        self.Q_table = np.zeros((self.TAILLE_ETAT, NB_ACTIONS))
        
        self.historique_recompenses = []
        self.meilleure_recompense = -float('inf')
        self.meilleur_chemin = []
        self.EPSILON = self.params.EPSILON_INIT
        self.MAX_STEPS_EPISODE = 100 # Valeur temporaire

    def reset_for_training(self):
        self.historique_recompenses = []
        self.Q_table = np.zeros((self.TAILLE_ETAT, NB_ACTIONS))
        self.EPSILON = self.params.EPSILON_INIT
        self.meilleure_recompense = -float('inf')
        self.meilleur_chemin = []
        
        # Ajustement dynamique de la durée de l'épisode selon la complexité
        etats_libres = self.TAILLE_ETAT - len(self.coords_obstacles)
        self.MAX_STEPS_EPISODE = max(50, int(etats_libres * 2.5))

    def train(self, progress_callback=None, live_path_callback=None):
        self.reset_for_training()
        ALPHA, GAMMA, EPISODES = self.params.ALPHA, self.params.GAMMA, self.params.EPISODES
        path_update_interval = max(1, EPISODES // 200)

        for episode in range(EPISODES):
            curr = self.coords_depart
            if curr is None: break
            
            idx_curr = coords_to_idx(curr, self.TAILLE_GRILLE_XY, self.NIVEAUX_ALTITUDE)
            done = False
            path = [curr]
            ep_reward = 0
            steps = 0
            
            while not done:
                # Epsilon-Greedy
                if random.random() < self.EPSILON: action = random.choice(list(ACTIONS.keys()))
                else:
                    max_q = np.max(self.Q_table[idx_curr, :])
                    actions = np.where(self.Q_table[idx_curr, :] == max_q)[0]
                    action = random.choice(actions) if len(actions) > 0 else random.choice(list(ACTIONS.keys()))
                
                # Action dans l'environnement (Stochastique activé)
                next_s, r, done_env = obtenir_etat_recompense_suivants(
                    curr, action, self.coords_cible, self.coords_obstacles, self.zones_vent, 
                    self.zones_thermiques, self.zones_descendantes, self.zones_inertie,
                    self.TAILLE_GRILLE_XY, self.NIVEAUX_ALTITUDE, mode_stochastique=True 
                )
                
                idx_next = coords_to_idx(next_s, self.TAILLE_GRILLE_XY, self.NIVEAUX_ALTITUDE)
                ep_reward += r
                path.append(next_s)
                
                # Mise à jour Q-Table (Bellman)
                old_q = self.Q_table[idx_curr, action]
                max_next_q = np.max(self.Q_table[idx_next, :]) if not done_env else 0
                self.Q_table[idx_curr, action] = (1 - ALPHA) * old_q + ALPHA * (r + GAMMA * max_next_q)
                
                curr = next_s
                idx_curr = idx_next
                steps += 1
                
                # Fin épisode (Cible ou Timeout)
                if steps > self.MAX_STEPS_EPISODE:
                    done = True
                    if curr != self.coords_cible: ep_reward += RECOMPENSE_COLLISION 
                else: done = done_env
            
            # Décroissance Epsilon
            if self.EPSILON > self.params.MIN_EPSILON: self.EPSILON *= self.params.DECAY_EPSILON
            self.historique_recompenses.append(ep_reward)
            
            # Sauvegarde meilleur
            if ep_reward > self.meilleure_recompense and curr == self.coords_cible:
                self.meilleure_recompense = ep_reward
                self.meilleur_chemin = list(path)
            
            # Callbacks UI
            if progress_callback and episode % 50 == 0: progress_callback(episode, self.meilleure_recompense)
            if live_path_callback and (episode + 1) % path_update_interval == 0: live_path_callback(path)
                
        return self.Q_table, self.meilleur_chemin, self.meilleure_recompense, self.historique_recompenses

def obtenir_chemin_optimal(Q_table, coords_depart, coords_cible, coords_obstacles, zones_vent, 
                           zones_thermiques, zones_descendantes, zones_inertie, taille_xy, niveaux_alt):
    """Joue le chemin optimal (Greedy) sans aléatoire."""
    if coords_depart is None or coords_cible is None: return []
    etats_libres = (taille_xy * taille_xy * niveaux_alt) - len(coords_obstacles)
    max_steps = int(etats_libres * 1.5) 
    curr = coords_depart
    path = [curr]
    steps = 0
    
    while curr != coords_cible and steps < max_steps:
        idx = coords_to_idx(curr, taille_xy, niveaux_alt)
        max_q = np.max(Q_table[idx, :])
        actions = np.where(Q_table[idx, :] == max_q)[0]
        if len(actions) == 0: break 
        action = random.choice(actions)
        
        # Mode Stochastique désactivé pour le replay
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

# --- 5. Interface Graphique (GUI) ---
class QLearningGUI:
    def __init__(self, master, agent, params):
        self.master = master
        self.agent = agent
        self.params = params
        master.title(f"Q-Learning Drone 3D | {TAILLE_GRILLE_XY}x{TAILLE_GRILLE_XY}x{self.agent.NIVEAUX_ALTITUDE}")
        
        self.CELL_SIZE = 40 
        self.canvas_width = self.CELL_SIZE * self.agent.TAILLE_GRILLE_XY
        self.canvas_height = self.CELL_SIZE * self.agent.TAILLE_GRILLE_XY
        
        self.current_path = []
        self.current_step = 0
        self.replay_running = False
        self.training_in_progress = False
        self.training_finished = False
        self.current_altitude_level = tk.IntVar(value=0)
        
        # --- Mise en page Notebook (Onglets) ---
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(pady=10, padx=10, fill="both", expand=True)

        self.frame_grid = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_grid, text="Éditeur & Replay")
        self.setup_grid_frame(self.frame_grid)

        self.frame_chart = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_chart, text="Apprentissage")
        self.setup_chart_frame(self.frame_chart)

        self.frame_stats = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_stats, text="Stats")
        self.setup_stats_frame(self.frame_stats)
        
        self.frame_3d = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_3d, text="Visualisation 3D")
        self.setup_3d_frame(self.frame_3d)
        
        self.frame_params = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_params, text="Paramètres")
        self.setup_params_frame(self.frame_params)
        
        self.canvas_grid.bind("<Button-1>", self.on_grid_click)
        self.draw_grid()
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

    # --- Onglet Paramètres ---
    def setup_params_frame(self, frame):
        # Variables existantes
        self.var_episodes = tk.IntVar(value=self.params.EPISODES)
        self.var_alpha = tk.DoubleVar(value=self.params.ALPHA)
        self.var_gamma = tk.DoubleVar(value=self.params.GAMMA)
        self.var_epsilon_init = tk.DoubleVar(value=self.params.EPSILON_INIT)
        
        # NOUVELLE VARIABLE : Altitude
        self.var_altitude = tk.IntVar(value=self.agent.NIVEAUX_ALTITUDE)
        
        main_frame = ttk.Frame(frame, padding="15")
        main_frame.pack(fill='both', expand=True)
        
        ttk.Label(main_frame, text="Paramètres Environnement", font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        
        # Champ Altitude
        row_alt = ttk.Frame(main_frame); row_alt.pack(fill='x', pady=5)
        ttk.Label(row_alt, text="Altitude Z (1 à 10):", width=20).pack(side=tk.LEFT)
        ttk.Entry(row_alt, textvariable=self.var_altitude, width=10).pack(side=tk.LEFT)
        ttk.Label(row_alt, text="(Affecte les éléments sur la map)", foreground="red", font=("Arial", 8)).pack(side=tk.LEFT, padx=5)

        ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=15)
        ttk.Label(main_frame, text="Hyperparamètres de l'Agent", font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        
        # Champs IA existants
        for txt, var in [("Episodes:", self.var_episodes), ("Alpha:", self.var_alpha), ("Gamma:", self.var_gamma), ("Epsilon Init:", self.var_epsilon_init)]:
            row = ttk.Frame(main_frame); row.pack(fill='x', pady=5)
            ttk.Label(row, text=txt, width=20).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=10).pack(side=tk.LEFT)
            
        ttk.Button(main_frame, text="Appliquer Tout", command=self.apply_params).pack(pady=20)
        self.lbl_advice = ttk.Label(main_frame, text="", foreground="blue"); self.lbl_advice.pack()
        self.update_advice()

    def apply_params(self):
        if self.training_in_progress: 
            messagebox.showwarning("Stop", "Impossible de modifier les paramètres pendant l'entraînement.")
            return

        # 1. Mise à jour des Hyperparamètres IA
        self.params.EPISODES = self.var_episodes.get()
        self.params.ALPHA = self.var_alpha.get()
        self.params.GAMMA = self.var_gamma.get()
        self.params.EPSILON_INIT = self.var_epsilon_init.get()
        self.progress_bar['maximum'] = self.params.EPISODES

        # 2. Gestion du changement d'Altitude
        new_alt = self.var_altitude.get()
        if new_alt < 1: new_alt = 1 # Sécurité
        if new_alt > 10: new_alt = 10 # Sécurité
        
        if new_alt != self.agent.NIVEAUX_ALTITUDE:
            # Mise à jour de l'agent
            self.agent.NIVEAUX_ALTITUDE = new_alt
            self.agent.TAILLE_ETAT = self.agent.TAILLE_GRILLE_XY * self.agent.TAILLE_GRILLE_XY * new_alt
            self.params.TAILLE_ETAT_BASE = self.agent.TAILLE_ETAT # Mise à jour param size
            
            # Reset complet de la mémoire de l'agent
            self.agent.reset_for_training()
            
            # Nettoyage de la map : on supprime les blocs qui sont désormais "hors limites" (trop haut)
            features = [self.agent.coords_obstacles, self.agent.zones_vent, self.agent.zones_thermiques, 
                        self.agent.zones_descendantes, self.agent.zones_inertie]
            for feature_list in features:
                # On garde seulement les éléments dont z < new_alt
                feature_list[:] = [coord for coord in feature_list if coord[2] < new_alt]
            
            # Reset Départ/Cible si hors limites
            if self.agent.coords_depart and self.agent.coords_depart[2] >= new_alt:
                self.agent.coords_depart = (self.agent.TAILLE_GRILLE_XY-1, 0, 0) # Reset sol
            if self.agent.coords_cible and self.agent.coords_cible[2] >= new_alt:
                self.agent.coords_cible = (0, self.agent.TAILLE_GRILLE_XY-1, new_alt-1) # Reset plafond
            
            # Mise à jour de l'UI
            self.current_altitude_level.set(0)
            self.scale_altitude.config(to=new_alt-1) # Mise à jour du slider
            self.draw_grid()
            self.draw_3d_environment()
            
            messagebox.showinfo("Mise à jour", f"Paramètres appliqués.\nL'environnement a été redimensionné à {new_alt} niveaux.")
        else:
            messagebox.showinfo("Succès", "Paramètres IA mis à jour.")
            
        self.update_advice()


    def update_advice(self):
        c = self.params.calculer_episodes_conseilles(self.agent.coords_obstacles)
        self.lbl_advice.config(text=f"Conseil IA : ~{c} épisodes recommandés")

    # --- Onglet Graphiques ---
    def setup_chart_frame(self, frame):
        self.fig_conv, self.ax_conv = plt.subplots(figsize=(5,4))
        self.canvas_chart = FigureCanvasTkAgg(self.fig_conv, master=frame)
        self.canvas_chart.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_stats_frame(self, frame):
        self.fig_stats, (self.ax_pos, self.ax_alt, self.ax_reward) = plt.subplots(3, 1, figsize=(5, 6), sharex=True)
        self.canvas_stats = FigureCanvasTkAgg(self.fig_stats, master=frame)
        self.canvas_stats.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- ÉDITEUR & GRILLE (UX Améliorée) ---
    def setup_grid_frame(self, frame):
        self.canvas_grid = tk.Canvas(frame, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas_grid.pack(side=tk.LEFT, padx=10, pady=10)

        ctrl = ttk.Frame(frame); ctrl.pack(side=tk.RIGHT, padx=10, pady=10, fill="y")
        
        # 1. Slider Altitude (MODIFIÉ)
        lf_visu = ttk.LabelFrame(ctrl, text="Altitude de vue (Couche)", padding=5)
        lf_visu.pack(fill='x', pady=5)
        
        # On stocke le widget dans self.scale_altitude pour pouvoir le mettre à jour
        self.scale_altitude = tk.Scale(lf_visu, from_=0, to=self.agent.NIVEAUX_ALTITUDE-1, orient=tk.HORIZONTAL, 
                 variable=self.current_altitude_level, command=self.on_altitude_change)
        self.scale_altitude.pack(fill='x')


        # 2. Palette d'Outils avec Légende Couleur
        lf_tools = ttk.LabelFrame(ctrl, text="Palette d'Outils", padding=5)
        lf_tools.pack(fill='x', pady=5)
        
        self.edit_mode = tk.StringVar(value="obstacle")
        
        # Helper pour créer une ligne de palette propre
        def add_tool(parent, text, mode, color):
            row = ttk.Frame(parent)
            row.pack(fill='x', pady=2, anchor='w')
            # Carré de couleur
            cv = tk.Canvas(row, width=15, height=15, bg=color, highlightthickness=1, highlightbackground="gray")
            cv.pack(side=tk.LEFT, padx=(0,5))
            # Radiobutton
            ttk.Radiobutton(row, text=text, variable=self.edit_mode, value=mode).pack(side=tk.LEFT)

        ttk.Label(lf_tools, text="Structure:", font=("Arial", 9, "bold")).pack(anchor='w', pady=(5,0))
        add_tool(lf_tools, "Départ (A)", "depart", COLORS["depart"])
        add_tool(lf_tools, "Cible (T)", "cible", COLORS["cible"])
        add_tool(lf_tools, "Obstacle", "obstacle", COLORS["obstacle"])
        
        ttk.Label(lf_tools, text="Physique:", font=("Arial", 9, "bold")).pack(anchor='w', pady=(5,0))
        add_tool(lf_tools, "Thermique (Monte)", "thermique", COLORS["thermique"])
        add_tool(lf_tools, "Descendant (Desc.)", "descendant", COLORS["descendant"])
        add_tool(lf_tools, "Vent (Pénalité)", "vent", COLORS["vent"])
        add_tool(lf_tools, "Inertie (Glisse)", "inertie", COLORS["inertie"])
        
        ttk.Separator(lf_tools, orient='horizontal').pack(fill='x', pady=5)
        ttk.Radiobutton(lf_tools, text="Gomme / Effacer", variable=self.edit_mode, value="effacer").pack(anchor='w')


        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=10)
        btn_frame = ttk.Frame(ctrl)
        btn_frame.pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="Exporter Map (JSON)", command=self.export_map).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="Importer Map (JSON)", command=self.import_map).pack(fill='x', pady=2)
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=10)


        # 3. Actions Entraînement
        self.btn_train = ttk.Button(ctrl, text="Lancer Entraînement", command=self.start_training_manually)
        self.btn_train.pack(fill='x', pady=10)
        
        self.btn_replay = ttk.Button(ctrl, text="Replay Optimal", command=self.start_replay, state=tk.DISABLED)
        self.btn_replay.pack(fill='x')
        
        self.progress_bar = ttk.Progressbar(ctrl, length=200, mode='determinate')
        self.progress_bar.pack(pady=10)
        self.lbl_status = ttk.Label(ctrl, text="Statut : Prêt.")
        self.lbl_status.pack()

    # --- VISUALISATION 3D (Tous les blocs) ---
    def setup_3d_frame(self, frame):
        self.fig_3d = plt.figure(figsize=(5, 4))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=frame)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.draw_3d_environment([])

    def draw_3d_environment(self, path=None):
        if not self.master.winfo_exists(): return
        self.ax_3d.clear()
        self.ax_3d.view_init(elev=35, azim=-50)
        
        # Configuration Grille
        max_c = self.agent.TAILLE_GRILLE_XY - 1
        max_a = self.agent.NIVEAUX_ALTITUDE - 1
        self.ax_3d.set_xlim(0, max_c+1); self.ax_3d.set_ylim(0, max_c+1); self.ax_3d.set_zlim(0, max_a+1)
        self.ax_3d.set_xlabel("X"); self.ax_3d.set_ylabel("Y"); self.ax_3d.set_zlabel("Alt")
        self.ax_3d.invert_yaxis()

        # Helper pour dessiner des cubes
        def draw_voxels(coord_list, color, alpha, label=None):
            if not coord_list: return
            for r, c, a in coord_list:
                self.ax_3d.bar3d(c, r, a, 1, 1, 1, color=color, alpha=alpha, shade=True, edgecolor='gray', linewidth=0.1)

        # Dessin de TOUS les blocs avec transparences différentes
        draw_voxels(self.agent.coords_obstacles, COLORS["obstacle"], 0.8)
        draw_voxels(self.agent.zones_thermiques, COLORS["thermique"], 0.3)
        draw_voxels(self.agent.zones_descendantes, COLORS["descendant"], 0.3)
        draw_voxels(self.agent.zones_vent, COLORS["vent"], 0.2)
        draw_voxels(self.agent.zones_inertie, COLORS["inertie"], 0.2)

        # Départ et Cible
        if self.agent.coords_depart:
            d = self.agent.coords_depart
            self.ax_3d.scatter(d[1]+0.5, d[0]+0.5, d[2]+0.5, c=COLORS["depart"], s=100, label="Départ")
        if self.agent.coords_cible:
            t = self.agent.coords_cible
            self.ax_3d.scatter(t[1]+0.5, t[0]+0.5, t[2]+0.5, c=COLORS["cible"], marker='*', s=200, label="Cible")

        # Chemin Optimal
        if path:
            xs = [p[1]+0.5 for p in path]
            ys = [p[0]+0.5 for p in path]
            zs = [p[2]+0.5 for p in path]
            self.ax_3d.plot(xs, ys, zs, c="blue", linewidth=2, label="Chemin", marker='o', markersize=3)

        self.canvas_3d.draw()

    # --- Gestion des Événements ---
    def on_tab_change(self, event):
        # Rafraichir la 3D quand on clique sur l'onglet
        if self.notebook.tab(self.notebook.select(), "text") == "Visualisation 3D":
            self.draw_3d_environment(getattr(self, 'final_optimal_path', []))

    def on_grid_click(self, event):
        """Gestion de l'édition de la map au clic souris."""
        if self.training_in_progress: return
        c, r = event.x // self.CELL_SIZE, event.y // self.CELL_SIZE
        a = self.current_altitude_level.get()
        
        if not (0 <= c < self.agent.TAILLE_GRILLE_XY and 0 <= r < self.agent.TAILLE_GRILLE_XY): return
        
        coords = (r, c, a)
        mode = self.edit_mode.get()
        
        # Nettoyage préalable de la case cliquée
        for lst in [self.agent.coords_obstacles, self.agent.zones_vent, self.agent.zones_thermiques, 
                    self.agent.zones_descendantes, self.agent.zones_inertie]:
            if coords in lst: lst.remove(coords)
        if coords == self.agent.coords_depart and mode != "depart": self.agent.coords_depart = None
        if coords == self.agent.coords_cible and mode != "cible": self.agent.coords_cible = None

        # Application du nouveau bloc selon le mode
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
        self.draw_grid(self.current_path if self.training_in_progress else getattr(self, 'final_optimal_path', None))

    def draw_grid(self, path=None):
        """Dessine la grille 2D pour la couche d'altitude sélectionnée."""
        self.canvas_grid.delete("all")
        sz = self.CELL_SIZE
        view_a = self.current_altitude_level.get()
        
        # 1. Dessin du fond (Map)
        for r in range(self.agent.TAILLE_GRILLE_XY):
            for c in range(self.agent.TAILLE_GRILLE_XY):
                x, y = c*sz, r*sz
                coord = (r, c, view_a)
                col, txt = COLORS["vide"], ""
                
                if coord in self.agent.zones_vent: col, txt = COLORS["vent"], "W"
                elif coord in self.agent.zones_thermiques: col, txt = COLORS["thermique"], "T"
                elif coord in self.agent.zones_descendantes: col, txt = COLORS["descendant"], "D"
                elif coord in self.agent.zones_inertie: col, txt = COLORS["inertie"], "I"
                elif coord in self.agent.coords_obstacles: col, txt = COLORS["obstacle"], ""
                
                if coord == self.agent.coords_depart: col, txt = COLORS["depart"], "A"
                if coord == self.agent.coords_cible: col, txt = COLORS["cible"], "T"
                
                self.canvas_grid.create_rectangle(x, y, x+sz, y+sz, fill=col, outline="gray")
                # Texte blanc si fond sombre
                text_col = "white" if col in [COLORS["obstacle"], COLORS["cible"]] else "black"
                if txt: self.canvas_grid.create_text(x+sz/2, y+sz/2, text=txt, fill=text_col, font=('Arial', 10, 'bold'))

        # 2. Dessin du Chemin (Path)
        if path:
            for i, (r, c, a) in enumerate(path):
                if a == view_a:
                    cx, cy = c*sz+sz/2, r*sz+sz/2
                    col = "orange" if i==len(path)-1 else "blue"
                    self.canvas_grid.create_oval(cx-3, cy-3, cx+3, cy+3, fill=col, outline=col)
                    
                    # Trait de liaison
                    if i>0:
                         pr, pc, pa = path[i-1]
                         if pa == a:
                             self.canvas_grid.create_line(pc*sz+sz/2, pr*sz+sz/2, cx, cy, fill="blue", width=2)
                         else:
                             # Indication changement altitude
                             txt = "⬆" if pa < a else "⬇"
                             self.canvas_grid.create_text(cx, cy, text=txt, fill="purple", font=('Arial', 14, 'bold'))
        
        # 3. Dessin de l'Agent (Position actuelle)
        pos = None
        if self.replay_running and path: pos = path[self.current_step]
        elif self.training_in_progress and path: pos = path[-1]
        elif self.agent.coords_depart: pos = self.agent.coords_depart
        
        if pos and pos[2] == view_a:
            cx, cy = pos[1]*sz+sz/2, pos[0]*sz+sz/2
            self.canvas_grid.create_oval(cx-8, cy-8, cx+8, cy+8, fill="red", outline="white", width=2)

    # --- Logique d'Entraînement ---
    def start_training_manually(self):
        if not self.agent.coords_depart or not self.agent.coords_cible:
            messagebox.showerror("Erreur", "Placez d'abord un point de Départ (A) et une Cible (T).")
            return
        
        self.training_in_progress = True
        self.btn_train.config(state=tk.DISABLED)
        self.btn_replay.config(state=tk.DISABLED)
        self.lbl_status['text'] = "Entraînement en cours..."
        
        threading.Thread(target=self.run_training, daemon=True).start()

    def run_training(self):
        self.agent.train(
            progress_callback=lambda ep, rew: self.master.after(0, self.update_progress, ep, rew),
            live_path_callback=lambda p: self.master.after(0, self.update_live, p)
        )
        self.master.after(0, self.finish_training)

    def update_progress(self, ep, rew):
        self.progress_bar['value'] = ep
        self.lbl_status['text'] = f"Ep: {ep} | Meilleure Récompense: {rew:.1f}"

    def update_live(self, path):
        self.current_path = path
        # Si on est sur l'onglet grille, on rafraichit la vue
        if self.notebook.tab(self.notebook.select(), "text") == "Éditeur & Replay":
            if path and path[-1][2] != self.current_altitude_level.get():
                self.current_altitude_level.set(path[-1][2]) # Suit l'altitude de l'agent
            self.draw_grid(path)

    def finish_training(self):
        self.training_in_progress = False
        self.btn_train.config(state=tk.NORMAL)
        self.btn_replay.config(state=tk.NORMAL)
        self.lbl_status['text'] = "Entraînement Terminé. Meilleure Récompense :", RECOMPENSE_CIBLE
        # Calcul du chemin final sans aléatoire
        self.final_optimal_path = obtenir_chemin_optimal(
            self.agent.Q_table, self.agent.coords_depart, self.agent.coords_cible,
            self.agent.coords_obstacles, self.agent.zones_vent,
            self.agent.zones_thermiques, self.agent.zones_descendantes, self.agent.zones_inertie,
            self.agent.TAILLE_GRILLE_XY, self.agent.NIVEAUX_ALTITUDE
        )
        
        self.draw_grid(self.final_optimal_path)
        self.draw_3d_environment(self.final_optimal_path)
        self.plot_graphs()
        messagebox.showinfo("Terminé", "L'entraînement est fini. Vous pouvez visualiser le replay ou la 3D.")

    # --- Replay ---
    def start_replay(self):
        if not self.final_optimal_path: return
        self.replay_running = True
        self.current_path = self.final_optimal_path
        self.current_step = 0
        self.btn_replay.config(state=tk.DISABLED)
        self.loop_replay()

    def loop_replay(self):
        if not self.replay_running: return
        
        if self.current_step < len(self.current_path):
            p = self.current_path[self.current_step]
            if p[2] != self.current_altitude_level.get(): self.current_altitude_level.set(p[2])
            
            self.draw_grid(self.current_path[:self.current_step+1])
            self.current_step += 1
            self.master.after(150, self.loop_replay)
        else:
            self.replay_running = False
            self.btn_replay.config(state=tk.NORMAL)
            self.lbl_status['text'] = "Replay terminé."
            
            
            
    def export_map(self):
        """Sauvegarde la configuration actuelle dans un JSON."""
        if self.training_in_progress: 
            messagebox.showwarning("Attention", "Impossible d'exporter pendant l'entraînement.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".json", 
            filetypes=[("Fichier JSON", "*.json")]
        )
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
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
            messagebox.showinfo("Succès", "La map a été exportée correctement.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'exportation :\n{e}")

    def import_map(self):
        """Charge une configuration depuis un JSON."""
        if self.training_in_progress: return

        filepath = filedialog.askopenfilename(filetypes=[("Fichier JSON", "*.json")])
        if not filepath: return

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Vérification de la compatibilité
            if (data["taille_xy"] != self.agent.TAILLE_GRILLE_XY or 
                data["niveaux_altitude"] != self.agent.NIVEAUX_ALTITUDE):
                messagebox.showerror("Erreur", "La map chargée ne correspond pas aux dimensions de la grille actuelle.")
                return

            # Chargement des données (conversion en tuples nécessaire pour les listes de coords)
            self.agent.coords_depart = tuple(data["depart"]) if data["depart"] else None
            self.agent.coords_cible = tuple(data["cible"]) if data["cible"] else None
            self.agent.coords_obstacles = [tuple(x) for x in data["obstacles"]]
            self.agent.zones_vent = [tuple(x) for x in data["vent"]]
            self.agent.zones_thermiques = [tuple(x) for x in data["thermiques"]]
            self.agent.zones_descendantes = [tuple(x) for x in data["descendants"]]
            self.agent.zones_inertie = [tuple(x) for x in data["inertie"]]
            
            # Mise à jour de l'affichage
            self.draw_grid()
            self.draw_3d_environment()
            self.update_advice()
            messagebox.showinfo("Succès", "Map importée avec succès.")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de lire le fichier :\n{e}")
    

    def plot_graphs(self):
        # --- 1. Graphique de Convergence (Apprentissage) ---
        self.ax_conv.clear()
        if self.agent.historique_recompenses:
            # A. Récompenses brutes (Gris transparent)
            self.ax_conv.plot(self.agent.historique_recompenses, color='gray', alpha=0.3, label="Brut", linewidth=0.8)

            # B. Moyenne Glissante (Rouge)
            w = 50
            if len(self.agent.historique_recompenses) > w:
                avg = np.convolve(self.agent.historique_recompenses, np.ones(w)/w, mode='valid')
                # On décale l'axe X pour aligner visuellement la moyenne
                self.ax_conv.plot(np.arange(w-1, len(self.agent.historique_recompenses)), avg, color='red', label="Moyenne", linewidth=2)
        
        self.ax_conv.set_title("Convergence de l'apprentissage")
        self.ax_conv.set_xlabel("Épisodes")
        self.ax_conv.set_ylabel("Récompense")
        self.ax_conv.legend()
        self.ax_conv.grid(True, alpha=0.3)
        self.canvas_chart.draw()
        
        # --- 2. Stats du chemin optimal ---
        p = self.final_optimal_path
        self.ax_pos.clear(); self.ax_alt.clear(); self.ax_reward.clear()
        
        if p and len(p) > 1:
            # --- Graph 1 : Vue en Plongée (Trajectoire 2D X/Y) ---
            cols = [x[1] for x in p] # Colonne (X)
            rows = [x[0] for x in p] # Ligne (Y)
            
            # Tracé du chemin (Vue de dessus)
            self.ax_pos.plot(cols, rows, marker='.', linestyle='-', color='blue', label='Trajet')
            
            # Départ (Rouge) et Arrivée (Vert)
            self.ax_pos.plot(cols[0], rows[0], marker='o', color='red', markersize=8, label='Départ')
            self.ax_pos.plot(cols[-1], rows[-1], marker='*', color='green', markersize=12, label='Arrivée')
            
            # Configuration "Carte"
            self.ax_pos.set_title("Trajectoire 2D (Vue de dessus)")
            self.ax_pos.set_xlabel("Colonnes (X)")
            self.ax_pos.set_ylabel("Lignes (Y)")
            self.ax_pos.invert_yaxis()    # Important : (0,0) en haut à gauche
            self.ax_pos.set_aspect('equal') # Garder les proportions carrées
            self.ax_pos.grid(True, linestyle=':')
            self.ax_pos.legend(loc='best', fontsize='small')

            # --- Graph 2 : Altitude (Z) vs Temps ---
            steps = range(len(p))
            self.ax_alt.plot(steps, [x[2] for x in p], color='green', marker='o')
            self.ax_alt.set_title("Altitude (Z)")
            self.ax_alt.set_xlabel("Étapes")
            self.ax_alt.set_ylabel("Niveau Z")
            self.ax_alt.set_yticks(range(self.agent.NIVEAUX_ALTITUDE))
            self.ax_alt.grid(True)
            
            # --- Graph 3 : Récompense Cumulée ---
            rewards = []
            cum = 0
            for i in range(len(p)-1):
                curr = p[i]
                nxt = p[i+1]
                # Retrouver l'action par déduction
                dr, dc, da = nxt[0]-curr[0], nxt[1]-curr[1], nxt[2]-curr[2]
                act = -1
                if dr == -1: act = 0
                elif dr == 1: act = 1
                elif dc == -1: act = 2
                elif dc == 1: act = 3
                elif da == 1: act = 4
                elif da == -1: act = 5
                
                if act != -1:
                    _, r, _ = obtenir_etat_recompense_suivants(
                        curr, act, self.agent.coords_cible, self.agent.coords_obstacles,
                        self.agent.zones_vent, self.agent.zones_thermiques, 
                        self.agent.zones_descendantes, self.agent.zones_inertie,
                        self.agent.TAILLE_GRILLE_XY, self.agent.NIVEAUX_ALTITUDE,
                        mode_stochastique=False
                    )
                    
                    if nxt == self.agent.coords_cible:
                        r -= RECOMPENSE_CIBLE
                    
                    cum += r
                    rewards.append(cum)
            
            if rewards:
                self.ax_reward.plot(rewards, color='purple')
                self.ax_reward.set_title("Score Cumulé")
                self.ax_reward.set_xlabel("Étapes")
                self.ax_reward.grid(True)

        self.fig_stats.tight_layout()
        self.canvas_stats.draw()


# --- POINT D'ENTRÉE MAIN ---
if __name__ == "__main__":
    # On définit l'altitude par défaut ici (ex: 3)
    # L'utilisateur pourra la changer ensuite dans l'onglet "Paramètres"
    DEFAULT_ALTITUDE = 3
    
    # Initialisation de l'application
    params = QLearningParameters(TAILLE_GRILLE_XY * TAILLE_GRILLE_XY * DEFAULT_ALTITUDE, NB_ACTIONS)
    agent = QLearningAgent(params, TAILLE_GRILLE_XY, DEFAULT_ALTITUDE)
    
    root = tk.Tk()
    
    # Configuration du style pour que ce soit un peu plus moderne
    style = ttk.Style()
    style.theme_use('clam')
    
    gui = QLearningGUI(root, agent, params)
    root.mainloop()
