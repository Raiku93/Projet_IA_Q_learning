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

# --- 1. Définition de l'environnement 3D (Drone) et Paramètres Statiques ---
TAILLE_GRILLE_XY = 10
NIVEAUX_ALTITUDE = 3
TAILLE_ETAT_BASE = TAILLE_GRILLE_XY * TAILLE_GRILLE_XY * NIVEAUX_ALTITUDE

ACTIONS = {
    0: 'HAUT_2D',    # (r-1, c, a)
    1: 'BAS_2D',     # (r+1, c, a)
    2: 'GAUCHE',     # (r, c-1, a)
    3: 'DROITE',     # (r, c+1, a)
    4: 'MONTER',     # (r, c, a+1)
    5: 'DESCENDRE'   # (r, c, a-1)
}
NB_ACTIONS = len(ACTIONS)

RECOMPENSE_CIBLE = 1000
RECOMPENSE_COLLISION = -50
RECOMPENSE_MOUVEMENT_2D = -1
RECOMPENSE_MOUVEMENT_3D = -3
RECOMPENSE_PENALITE_VENT = -4
# EPISODES = 20000 -> Déplacé dans la classe de paramètres

# --- Classe pour gérer les paramètres (NOUVEAU) ---
class QLearningParameters:
    def __init__(self, taille_etat_base, nb_actions):
        self.TAILLE_ETAT_BASE = taille_etat_base
        self.NB_ACTIONS = nb_actions
        
        # Hyperparamètres par défaut
        self.EPISODES = 20000
        self.ALPHA = 0.1  # Taux d'apprentissage
        self.GAMMA = 0.9  # Facteur d'actualisation
        self.EPSILON_INIT = 1.0 # Epsilon initial
        self.DECAY_EPSILON = 0.9998 
        self.MIN_EPSILON = 0.01

    def calculer_episodes_conseilles(self, coords_obstacles):
        """Calcule un nombre d'épisodes conseillé basé sur la complexité."""
        nombre_etats_libres = self.TAILLE_ETAT_BASE - len(coords_obstacles)
        
        # Algorithme simple : plus il y a d'états libres, plus il faut d'épisodes.
        # Facteur de base multiplié par la taille de l'espace.
        facteur_complexite = 1.0 # 1.0 pour une grille simple, ajuster si besoin
        
        # Min 5000, Max 50000. Utilise l'heuristique (états libres * 200)
        conseil = int(nombre_etats_libres * 200 * facteur_complexite)
        
        return max(5000, min(22000, conseil))

# --- Fonctions de l'Environnement 3D (restent inchangées) ---
def coords_to_idx(coords, taille_xy, niveaux_altitude):
    r, c, a = coords
    taille_couche_xy = taille_xy * taille_xy
    return a * taille_couche_xy + r * taille_xy + c

def initialiser_positions_3d(taille_xy, niveaux_alt):
    coords_depart = (taille_xy - 1, 0, 0)
    coords_cible = (0, taille_xy - 1, niveaux_alt - 1) # Modifié pour être toujours au niveau le plus haut
    coords_obstacles = []
    zones_vent = []
    
    for r in range(taille_xy // 4, taille_xy // 2):
        for a in range(niveaux_alt - 1): 
            coords_obstacles.append((r, taille_xy // 3, a))
            
    for r in range(taille_xy // 2, 3 * taille_xy // 4):
         for a in range(niveaux_alt): 
             coords_obstacles.append((r, 2 * taille_xy // 3, a))

    for a_vent in [niveaux_alt - 2, niveaux_alt - 1]:
        for r in range(taille_xy // 2, taille_xy):
            for c in range(0, taille_xy // 2):
                coords_vent = (r, c, a_vent)
                if coords_vent not in coords_obstacles and coords_vent != coords_cible:
                    zones_vent.append(coords_vent)
                    
    if coords_depart in coords_obstacles: coords_obstacles.remove(coords_depart)
    if coords_cible in coords_obstacles: coords_obstacles.remove(coords_cible)

    return coords_depart, coords_cible, coords_obstacles, zones_vent

def obtenir_etat_recompense_suivants(etat_courant, action_index, coords_cible, coords_obstacles, zones_vent, taille_xy, niveaux_alt):
    r, c, a = etat_courant
    new_r, new_c, new_a = r, c, a

    if action_index == 0: new_r -= 1
    elif action_index == 1: new_r += 1
    elif action_index == 2: new_c -= 1
    elif action_index == 3: new_c += 1
    elif action_index == 4: new_a += 1
    elif action_index == 5: new_a -= 1

    nouvel_etat = (new_r, new_c, new_a)

    if not (0 <= new_r < taille_xy and 0 <= new_c < taille_xy and 0 <= new_a < niveaux_alt):
        return etat_courant, RECOMPENSE_COLLISION, True 
    if nouvel_etat in coords_obstacles:
        return etat_courant, RECOMPENSE_COLLISION, True 
    if nouvel_etat == coords_cible:
        return nouvel_etat, RECOMPENSE_CIBLE, True 
    
    recompense = 0
    if action_index in [0, 1, 2, 3]: 
        recompense = RECOMPENSE_MOUVEMENT_2D
    elif action_index in [4, 5]: 
        recompense = RECOMPENSE_MOUVEMENT_3D
        
    if nouvel_etat in zones_vent:
        recompense += RECOMPENSE_PENALITE_VENT
    
    return nouvel_etat, recompense, False 

# --- 2. Classe QLearningAgent (MODIFIÉE) ---
class QLearningAgent:
    # L'agent prend maintenant l'instance de paramètres
    def __init__(self, params: QLearningParameters, taille_xy, niveaux_altitude):
        self.params = params
        self.TAILLE_GRILLE_XY = taille_xy
        self.NIVEAUX_ALTITUDE = niveaux_altitude
        
        self.coords_depart, self.coords_cible, self.coords_obstacles, self.zones_vent = \
            initialiser_positions_3d(self.TAILLE_GRILLE_XY, self.NIVEAUX_ALTITUDE)
            
        self.TAILLE_ETAT = self.TAILLE_GRILLE_XY * self.TAILLE_GRILLE_XY * self.NIVEAUX_ALTITUDE
        self.Q_table = np.zeros((self.TAILLE_ETAT, NB_ACTIONS))
        
        self.historique_recompenses = []
        self.meilleure_recompense = -float('inf')
        self.meilleur_chemin = []
        self.EPSILON = self.params.EPSILON_INIT # Initialisation EPSILON

        nombre_etats_libres = self.TAILLE_ETAT - len(self.coords_obstacles)
        facteur_allongement = 2.5 
        self.MAX_STEPS_EPISODE = int(nombre_etats_libres * facteur_allongement)
        
        min_steps = self.TAILLE_GRILLE_XY * self.NIVEAUX_ALTITUDE * 2
        if self.MAX_STEPS_EPISODE < min_steps: 
             self.MAX_STEPS_EPISODE = min_steps
        
    def reset_for_training(self):
        """Réinitialise l'état de l'agent pour un nouvel entraînement."""
        self.historique_recompenses = []
        self.Q_table = np.zeros((self.TAILLE_ETAT, NB_ACTIONS))
        self.EPSILON = self.params.EPSILON_INIT
        self.meilleure_recompense = -float('inf')
        self.meilleur_chemin = []


    # Ajout d'un callback pour la visualisation en direct (MODIFIÉ)
    def train(self, progress_callback=None, live_path_callback=None):
        self.reset_for_training()
        
        # Variables locales pour les paramètres pour éviter d'accéder constamment à self.params
        ALPHA = self.params.ALPHA
        GAMMA = self.params.GAMMA
        DECAY_EPSILON = self.params.DECAY_EPSILON
        MIN_EPSILON = self.params.MIN_EPSILON
        EPISODES = self.params.EPISODES
        
        # Contrôle de la fréquence de mise à jour du chemin en direct
        path_update_interval = max(1, EPISODES // 200) # Mise à jour environ 200 fois max

        for episode in range(EPISODES):
            etat_coords_courant = self.coords_depart
            
            if etat_coords_courant is None:
                print("Erreur : Point de départ non défini.")
                break
                
            etat_idx_courant = coords_to_idx(etat_coords_courant, self.TAILLE_GRILLE_XY, self.NIVEAUX_ALTITUDE)
            termine = False
            chemin_episode = [etat_coords_courant]
            recompense_episode = 0
            steps = 0
            
            while not termine:
                if random.uniform(0, 1) < self.EPSILON:
                    action_index = random.choice(list(ACTIONS.keys()))
                else:
                    max_q_value = np.max(self.Q_table[etat_idx_courant, :])
                    meilleures_actions = np.where(self.Q_table[etat_idx_courant, :] == max_q_value)[0]
                    if len(meilleures_actions) == 0:
                        action_index = random.choice(list(ACTIONS.keys())) 
                    else:
                        action_index = random.choice(meilleures_actions)
                    
                etat_coords_suivant, recompense, termine_env = obtenir_etat_recompense_suivants(
                    etat_coords_courant, action_index, self.coords_cible, 
                    self.coords_obstacles, self.zones_vent, 
                    self.TAILLE_GRILLE_XY, self.NIVEAUX_ALTITUDE
                )
                
                etat_idx_suivant = coords_to_idx(etat_coords_suivant, self.TAILLE_GRILLE_XY, self.NIVEAUX_ALTITUDE)
                recompense_episode += recompense
                chemin_episode.append(etat_coords_suivant)
                
                q_value_ancien = self.Q_table[etat_idx_courant, action_index]
                max_q_suivant = np.max(self.Q_table[etat_idx_suivant, :]) if not termine_env else 0
                # Utilise self.params.ALPHA et self.params.GAMMA
                q_value_nouveau = (1 - ALPHA) * q_value_ancien + ALPHA * (recompense + GAMMA * max_q_suivant)
                self.Q_table[etat_idx_courant, action_index] = q_value_nouveau
                
                etat_coords_courant = etat_coords_suivant
                etat_idx_courant = etat_idx_suivant
                steps += 1
                
                if steps > self.MAX_STEPS_EPISODE:
                    termine = True
                    if etat_coords_courant != self.coords_cible:
                        recompense_episode += RECOMPENSE_COLLISION 
                else:
                    termine = termine_env
                    
            # Utilise self.params.DECAY_EPSILON et self.params.MIN_EPSILON
            if self.EPSILON > MIN_EPSILON:
                self.EPSILON *= DECAY_EPSILON

            self.historique_recompenses.append(recompense_episode)
            
            if recompense_episode > self.meilleure_recompense and etat_coords_courant == self.coords_cible:
                self.meilleure_recompense = recompense_episode
                self.meilleur_chemin = list(chemin_episode)
            
            # Callback pour la progression de l'entraînement
            if progress_callback:
                progress_callback(episode + 1, self.meilleure_recompense)
                
            # Callback pour la visualisation en direct (NOUVEAU)
            if live_path_callback and (episode + 1) % path_update_interval == 0:
                 live_path_callback(chemin_episode)
                
        return self.Q_table, self.meilleur_chemin, self.meilleure_recompense, self.historique_recompenses


# --- 3. Fonction obtenir_chemin_optimal (inchangée) ---
def obtenir_chemin_optimal(Q_table, coords_depart, coords_cible, coords_obstacles, zones_vent, taille_xy, niveaux_alt):
    
    if coords_depart is None or coords_cible is None:
        return []

    taille_etat = taille_xy * taille_xy * niveaux_alt
    nombre_etats_libres = taille_etat - len(coords_obstacles)
    max_etapes = int(nombre_etats_libres * 1.5) 

    etat_coords_courant = coords_depart
    chemin = [etat_coords_courant]
    etapes = 0
    
    while etat_coords_courant != coords_cible and etapes < max_etapes:
        etat_idx_courant = coords_to_idx(etat_coords_courant, taille_xy, niveaux_alt)
        
        max_q_value = np.max(Q_table[etat_idx_courant, :])
        meilleures_actions = np.where(Q_table[etat_idx_courant, :] == max_q_value)[0]
        
        if len(meilleures_actions) == 0:
            break 
        action_index = random.choice(meilleures_actions)
        
        etat_coords_suivant, _, termine = obtenir_etat_recompense_suivants(
            etat_coords_courant, action_index, coords_cible, coords_obstacles, zones_vent, taille_xy, niveaux_alt
        )
        
        if termine and etat_coords_suivant != coords_cible:
            break

        chemin.append(etat_coords_suivant)
        etat_coords_courant = etat_coords_suivant
        etapes += 1
        
    return chemin

# --- 4. Interface Graphique (GUI) (MODIFIÉE POUR LES PARAMÈTRES ET LE LIVE PATH) ---

class QLearningGUI:
    def __init__(self, master, agent, params):
        self.master = master
        self.agent = agent
        self.params = params # Référence aux paramètres
        master.title(f"Q-Learning Agent - Drone 3D ({TAILLE_GRILLE_XY}x{TAILLE_GRILLE_XY}x{NIVEAUX_ALTITUDE})")
        
        self.CELL_SIZE = 40 
        self.canvas_width = self.CELL_SIZE * self.agent.TAILLE_GRILLE_XY
        self.canvas_height = self.CELL_SIZE * self.agent.TAILLE_GRILLE_XY
        
        self.current_path = []
        self.current_step = 0
        self.replay_running = False
        self.training_in_progress = False
        self.training_finished = False
        
        self.current_altitude_level = tk.IntVar(value=self.agent.coords_depart[2] if self.agent.coords_depart else 0)
        self.final_optimal_path = [] 

        self.notebook = ttk.Notebook(master)
        self.notebook.pack(pady=10, padx=10, fill="both", expand=True)

        # 1. Éditeur/Replay 2D
        self.frame_grid = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_grid, text="Éditeur de Grille & Replay (2D)")
        self.setup_grid_frame(self.frame_grid)

        # 2. Historique d'Apprentissage
        self.frame_chart = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_chart, text="Historique d'Apprentissage")
        self.setup_chart_frame(self.frame_chart)

        # 3. Statistiques du Chemin
        self.frame_stats = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_stats, text="Statistiques du Chemin")
        self.setup_stats_frame(self.frame_stats)
        
        # 4. Visualisation 3D Isométrique
        self.frame_3d = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_3d, text="Visualisation 3D Isométrique")
        self.setup_3d_frame(self.frame_3d)
        
        # 5. Paramètres Q-Learning (NOUVEAU)
        self.frame_params = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_params, text="Paramètres Q-Learning")
        self.setup_params_frame(self.frame_params) # NOUVEL APPEL
        
        self.canvas_grid.bind("<Button-1>", self.on_grid_click)
        self.draw_grid()
        
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

    # --- NOUVEAU : Configuration de l'onglet Paramètres ---
    def setup_params_frame(self, frame):
        """Configure les champs d'édition pour les hyperparamètres."""
        
        # Variables de contrôle pour les Entry
        self.var_episodes = tk.IntVar(value=self.params.EPISODES)
        self.var_alpha = tk.DoubleVar(value=self.params.ALPHA)
        self.var_gamma = tk.DoubleVar(value=self.params.GAMMA)
        self.var_epsilon_init = tk.DoubleVar(value=self.params.EPSILON_INIT)
        self.var_decay = tk.DoubleVar(value=self.params.DECAY_EPSILON)
        self.var_min_epsilon = tk.DoubleVar(value=self.params.MIN_EPSILON)
        
        main_frame = ttk.Frame(frame, padding="15")
        main_frame.pack(fill='both', expand=True)
        
        # Titre
        ttk.Label(main_frame, text="Hyperparamètres de l'Agent Q-Learning", font=('Arial', 14, 'bold')).pack(pady=(0, 50))
        
        # Création des champs de saisie
        fields = [
            ("Nombre d'Épisodes (EPISODES):", self.var_episodes),
            ("Taux d'Apprentissage (ALPHA):", self.var_alpha),
            ("Facteur d'Actualisation (GAMMA):", self.var_gamma),
            ("Epsilon Initial (EPSILON_INIT):", self.var_epsilon_init),
            ("Taux de Décroissance Epsilon (DECAY_EPSILON):", self.var_decay),
            ("Epsilon Minimum (MIN_EPSILON):", self.var_min_epsilon),
        ]
        
        for label_text, var in fields:
            row_frame = ttk.Frame(main_frame)
            row_frame.pack(fill='x', pady=5)
            
            ttk.Label(row_frame, text=label_text, width=30).pack(side=tk.LEFT)
            entry = ttk.Entry(row_frame, textvariable=var, width=15)
            entry.pack(side=tk.LEFT, padx=10)
        
        ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=20)
        
        # Zone des conseils et bouton d'application
        
        # Conseil en Épisodes
        self.label_conseil = ttk.Label(main_frame, text="", foreground="blue")
        self.label_conseil.pack(pady=5, anchor='w')
        self.update_episode_advice()
        
        # Bouton Appliquer
        self.button_apply = ttk.Button(main_frame, text="Appliquer les Paramètres", command=self.apply_parameters)
        self.button_apply.pack(pady=15)
        
        # Bouton Réinitialiser
        self.button_reset = ttk.Button(main_frame, text="Réinitialiser aux Valeurs par Défaut", command=self.reset_parameters)
        self.button_reset.pack(pady=5)
        
        self.label_status_params = ttk.Label(main_frame, text="Statut: Paramètres par défaut.", foreground="red")
        self.label_status_params.pack(pady=10)


    def update_episode_advice(self):
        """Met à jour l'affichage du conseil en épisodes."""
        conseil = self.params.calculer_episodes_conseilles(self.agent.coords_obstacles)
        self.label_conseil.config(text=f"Conseil : 10000 pour simple, 20000 pour normal, 30000 pour complexe.")

    def apply_parameters(self):
        """Applique les valeurs des Entry à l'objet params de l'agent."""
        if self.training_in_progress:
            messagebox.showwarning("Action impossible", "Veuillez attendre la fin de l'entraînement pour modifier les paramètres.")
            return

        try:
            # Récupération et validation
            self.params.EPISODES = max(1, self.var_episodes.get())
            self.params.ALPHA = max(0.0, min(1.0, self.var_alpha.get()))
            self.params.GAMMA = max(0.0, min(1.0, self.var_gamma.get()))
            self.params.EPSILON_INIT = max(0.0, min(1.0, self.var_epsilon_init.get()))
            self.params.DECAY_EPSILON = max(0.0, min(1.0, self.var_decay.get()))
            self.params.MIN_EPSILON = max(0.0, min(1.0, self.var_min_epsilon.get()))
            
            # Mise à jour de la barre de progression
            self.progress_bar['maximum'] = self.params.EPISODES
            self.label_episode['text'] = f"Episode: 0 / {self.params.EPISODES}"
            
            self.training_finished = False # Les nouveaux paramètres impliquent un nouvel entraînement
            self.button_replay.config(state=tk.DISABLED)
            
            self.label_status_params.config(text="Statut: Paramètres appliqués. Prêt pour l'entraînement.", foreground="green")
            messagebox.showinfo("Succès", "Les paramètres ont été mis à jour.")
            
        except tk.TclError:
            messagebox.showerror("Erreur de Saisie", "Veuillez entrer des nombres valides pour tous les champs.")

    def reset_parameters(self):
        """Réinitialise les paramètres aux valeurs par défaut."""
        if self.training_in_progress:
            messagebox.showwarning("Action impossible", "Veuillez attendre la fin de l'entraînement.")
            return

        # Créer une nouvelle instance par défaut (ou réinitialiser les valeurs de la classe)
        default_params = QLearningParameters(self.params.TAILLE_ETAT_BASE, self.params.NB_ACTIONS)
        self.params.__dict__.update(default_params.__dict__) # Copie des attributs
        
        # Mise à jour des variables de l'interface
        self.var_episodes.set(self.params.EPISODES)
        self.var_alpha.set(self.params.ALPHA)
        self.var_gamma.set(self.params.GAMMA)
        self.var_epsilon_init.set(self.params.EPSILON_INIT)
        self.var_decay.set(self.params.DECAY_EPSILON)
        self.var_min_epsilon.set(self.params.MIN_EPSILON)
        
        self.apply_parameters() # Appliquer pour mettre à jour la barre de progression
        self.label_status_params.config(text="Statut: Paramètres réinitialisés et appliqués.", foreground="blue")

    # --- Fin de la section Paramètres ---
    
    def setup_grid_frame(self, frame):
        self.canvas_grid = tk.Canvas(frame, width=self.canvas_width, height=self.canvas_height, bg="white", borderwidth=0, highlightthickness=0)
        self.canvas_grid.pack(side=tk.LEFT, padx=10, pady=10)

        frame_controls = ttk.Frame(frame)
        frame_controls.pack(side=tk.RIGHT, padx=10, pady=10, fill="y", anchor="n")
        
        # --- Visualisation 2D ---
        ttk.Label(frame_controls, text="VISUALISATION 2D", font=('Arial', 12, 'bold')).pack(pady=(5,0), anchor="w")
        self.altitude_slider = tk.Scale(
            frame_controls, 
            from_=0, 
            to=self.agent.NIVEAUX_ALTITUDE - 1, 
            orient=tk.HORIZONTAL, 
            label="Niveau d'Altitude (A):", 
            variable=self.current_altitude_level, 
            command=self.on_altitude_change,
            length=250
        )
        self.altitude_slider.pack(pady=5, fill='x')
        
        ttk.Separator(frame_controls, orient='horizontal').pack(fill='x', pady=10)

        # --- Éditeur ---
        ttk.Label(frame_controls, text="ÉDITEUR", font=('Arial', 12, 'bold')).pack(pady=(5,0), anchor="w")
        self.edit_mode = tk.StringVar(value="obstacle")
        self.editor_frame = ttk.Frame(frame_controls)
        
        ttk.Radiobutton(self.editor_frame, text="Obstacle (X)", variable=self.edit_mode, value="obstacle").pack(anchor="w")
        ttk.Radiobutton(self.editor_frame, text="Vent (W)", variable=self.edit_mode, value="vent").pack(anchor="w")
        ttk.Radiobutton(self.editor_frame, text="Départ (A)", variable=self.edit_mode, value="depart").pack(anchor="w")
        ttk.Radiobutton(self.editor_frame, text="Cible (T)", variable=self.edit_mode, value="cible").pack(anchor="w")
        ttk.Radiobutton(self.editor_frame, text="Effacer", variable=self.edit_mode, value="effacer").pack(anchor="w")
        
        self.editor_frame.pack(fill='x', pady=5)
        
        # --- Section Gestion Map ---
        ttk.Separator(frame_controls, orient='horizontal').pack(fill='x', pady=10)
        ttk.Label(frame_controls, text="GESTION MAP", font=('Arial', 12, 'bold')).pack(pady=(5,0), anchor="w")
        
        self.button_export_map = ttk.Button(frame_controls, text="Exporter la Map...", command=self.export_map)
        self.button_export_map.pack(pady=5, fill='x')
        self.button_import_map = ttk.Button(frame_controls, text="Importer la Map...", command=self.import_map)
        self.button_import_map.pack(pady=5, fill='x')
        
        ttk.Separator(frame_controls, orient='horizontal').pack(fill='x', pady=10)

        # --- Contrôle ---
        ttk.Label(frame_controls, text="CONTRÔLE", font=('Arial', 12, 'bold')).pack(pady=5, anchor="w")
        
        self.button_start_train = ttk.Button(frame_controls, text="Lancer l'entraînement", command=self.start_training_manually)
        self.button_start_train.pack(pady=5, fill='x')
        
        self.button_replay = ttk.Button(frame_controls, text="Lancer Replay Optimal", command=self.start_replay, state=tk.DISABLED)
        self.button_replay.pack(pady=5, fill='x')
        
        self.label_replay_status = ttk.Label(frame_controls, text="Statut: Prêt pour l'édition.")
        self.label_replay_status.pack(pady=2, anchor="w")
        
        ttk.Separator(frame_controls, orient='horizontal').pack(fill='x', pady=10)
        
        # --- Progression ---
        ttk.Label(frame_controls, text="PROGRESSION", font=('Arial', 12, 'bold')).pack(pady=5, anchor="w")
        self.progress_bar = ttk.Progressbar(frame_controls, orient='horizontal', length=250, mode='determinate', maximum=self.params.EPISODES)
        self.progress_bar.pack(pady=5, fill='x')
        self.label_episode = ttk.Label(frame_controls, text=f"Episode: 0 / {self.params.EPISODES}")
        self.label_episode.pack(pady=2, anchor="w")
        self.label_best_reward = ttk.Label(frame_controls, text="Meilleure Récompense: N/A")
        self.label_best_reward.pack(pady=2, anchor="w")

    def setup_chart_frame(self, frame):
        self.fig_conv, self.ax_conv = plt.subplots(figsize=(10, 6))
        self.canvas_chart = FigureCanvasTkAgg(self.fig_conv, master=frame)
        self.canvas_widget = self.canvas_chart.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.plot_convergence() 

    def setup_stats_frame(self, frame):
        self.fig_stats, (self.ax_pos, self.ax_alt, self.ax_reward) = plt.subplots(
            3, 1, figsize=(10, 8), sharex=True
        )
        
        self.canvas_stats = FigureCanvasTkAgg(self.fig_stats, master=frame)
        self.canvas_stats_widget = self.canvas_stats.get_tk_widget()
        self.canvas_stats_widget.pack(fill=tk.BOTH, expand=True)
        
        self.plot_path_statistics() 
        
    def setup_3d_frame(self, frame):
        """Configure le cadre pour la visualisation 3D Matplotlib."""
        self.fig_3d = plt.figure(figsize=(10, 8))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=frame)
        self.canvas_3d_widget = self.canvas_3d.get_tk_widget()
        self.canvas_3d_widget.pack(fill=tk.BOTH, expand=True)
        
        self.draw_3d_environment([])

    def on_tab_change(self, event):
        """Déclenche le dessin 3D lors de la sélection de l'onglet 3D."""
        selected_tab = self.notebook.tab(self.notebook.select(), "text")
        if selected_tab == "Visualisation 3D Isométrique":
            path_to_draw = getattr(self, 'final_optimal_path', [])
            self.draw_3d_environment(path_to_draw)
            
    # --- Fonction 3D (inchangée) ---
    def draw_3d_environment(self, path=None):
        """Dessine l'environnement 3D (grille, obstacles, chemin) avec une vue isométrique."""
        if not self.master.winfo_exists(): return
        
        self.ax_3d.clear()
        
        self.ax_3d.view_init(elev=30, azim=-60) 
        
        max_coord = self.agent.TAILLE_GRILLE_XY - 1
        max_alt = self.agent.NIVEAUX_ALTITUDE - 1
        
        self.ax_3d.set_title("Visualisation 3D Isométrique de l'Environnement et du Chemin")
        self.ax_3d.set_xlabel("Colonne (X)")
        self.ax_3d.set_ylabel("Ligne (Y)")
        self.ax_3d.set_zlabel("Altitude (Z)")
        
        self.ax_3d.set_xlim(0, max_coord + 1) 
        self.ax_3d.set_ylim(0, max_coord + 1)
        self.ax_3d.set_zlim(0, max_alt + 1)
        
        self.ax_3d.set_xticks(np.arange(0.5, self.agent.TAILLE_GRILLE_XY, 1), 
                              labels=np.arange(0, self.agent.TAILLE_GRILLE_XY, 1))
        self.ax_3d.set_yticks(np.arange(0.5, self.agent.TAILLE_GRILLE_XY, 1), 
                              labels=np.arange(0, self.agent.TAILLE_GRILLE_XY, 1))
        self.ax_3d.set_zticks(np.arange(0.5, self.agent.NIVEAUX_ALTITUDE, 1), 
                              labels=np.arange(0, self.agent.NIVEAUX_ALTITUDE, 1))

        self.ax_3d.invert_yaxis() 
        
        # --- 1. Dessin des Obstacles (Blocs) ---
        for r, c, a in self.agent.coords_obstacles:
            self.ax_3d.bar3d(c, r, a, 1, 1, 1, 
                             color='black', alpha=0.6, shade=True)

        # --- 2. Dessin des Zones de Vent ---
        for r, c, a in self.agent.zones_vent:
             self.ax_3d.bar3d(c, r, a, 1, 1, 1, 
                              color='lightblue', alpha=0.05, shade=False)
                              
        # --- 3. Dessin des Points Clés ---
        
        if self.agent.coords_depart:
            r_d, c_d, a_d = self.agent.coords_depart
            self.ax_3d.scatter(c_d + 0.5, r_d + 0.5, a_d + 0.5, 
                               color='red', marker='o', s=100, label='Départ (A)', depthshade=False)
                               
        if self.agent.coords_cible:
            r_t, c_t, a_t = self.agent.coords_cible
            self.ax_3d.scatter(c_t + 0.5, r_t + 0.5, a_t + 0.5, 
                               color='green', marker='*', s=300, label='Cible (T)', depthshade=False)

        # --- 4. Dessin du Chemin (Path) ---
        if path:
            path_x = [p[1] + 0.5 for p in path] # Col + 0.5
            path_y = [p[0] + 0.5 for p in path] # Row + 0.5
            path_z = [p[2] + 0.5 for p in path] # Alt + 0.5 (Centre du cube)
            
            # Dessiner la ligne du chemin
            self.ax_3d.plot(path_x, path_y, path_z, 
                            color='blue', linewidth=3, label='Chemin Optimal', marker='o')
                            
        self.ax_3d.legend(loc='lower left')
        self.fig_3d.tight_layout()
        self.canvas_3d.draw()
    # --- Fin Fonction 3D ---


    def export_map(self):
        """Sauvegarde l'environnement actuel dans un fichier JSON."""
        if self.training_in_progress:
            messagebox.showwarning("Action impossible", "Vous ne pouvez pas exporter pendant l'entraînement.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Exporter la Map",
            defaultextension=".json",
            filetypes=[("Fichier Map JSON", "*.json"), ("Tous les fichiers", "*.*")]
        )
        
        if not filepath:
            return 

        map_data = {
            "taille_xy": self.agent.TAILLE_GRILLE_XY,
            "niveaux_altitude": self.agent.NIVEAUX_ALTITUDE,
            "depart": self.agent.coords_depart,
            "cible": self.agent.coords_cible,
            "obstacles": self.agent.coords_obstacles,
            "vent": self.agent.zones_vent
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(map_data, f, indent=4)
            messagebox.showinfo("Exportation réussie", f"Map sauvegardée dans {filepath}")
        except Exception as e:
            messagebox.showerror("Erreur d'exportation", f"Impossible de sauvegarder le fichier:\n{e}")

    def import_map(self):
        """Charge un environnement depuis un fichier JSON."""
        if self.training_in_progress or self.training_finished:
            messagebox.showwarning("Action impossible", "Vous ne pouvez importer une map qu'avant de lancer l'entraînement.")
            return

        filepath = filedialog.askopenfilename(
            title="Importer la Map",
            filetypes=[("Fichier Map JSON", "*.json"), ("Tous les fichiers", "*.*")]
        )

        if not filepath:
            return 

        try:
            with open(filepath, 'r') as f:
                map_data = json.load(f)

            required_keys = ["taille_xy", "niveaux_altitude", "depart", "cible", "obstacles", "vent"]
            if not all(key in map_data for key in required_keys):
                raise KeyError("Le fichier JSON ne contient pas les clés requises pour la map.")

            if (map_data["taille_xy"] != self.agent.TAILLE_GRILLE_XY or
                map_data["niveaux_altitude"] != self.agent.NIVEAUX_ALTITUDE):
                messagebox.showerror("Erreur d'importation", 
                    f"La map est pour une grille {map_data['taille_xy']}x{map_data['taille_xy']}x{map_data['niveaux_altitude']}.\n"
                    f"Cette application est configurée pour {self.agent.TAILLE_GRILLE_XY}x{self.agent.TAILLE_GRILLE_XY}x{self.agent.NIVEAUX_ALTITUDE}.")
                return

            self.agent.coords_depart = tuple(map_data["depart"])
            self.agent.coords_cible = tuple(map_data["cible"])
            self.agent.coords_obstacles = [tuple(obs) for obs in map_data["obstacles"]]
            self.agent.zones_vent = [tuple(vent) for vent in map_data["vent"]]
            
            self.current_altitude_level.set(self.agent.coords_depart[2]) 
            self.draw_grid()
            self.update_episode_advice() # Mise à jour du conseil après l'importation
            messagebox.showinfo("Importation réussie", "La map a été chargée.")

        except json.JSONDecodeError:
            messagebox.showerror("Erreur d'importation", "Le fichier n'est pas un JSON valide.")
        except Exception as e:
            messagebox.showerror("Erreur d'importation", f"Une erreur est survenue:\n{e}")

    def clear_environment_for_editor(self):
        """Réinitialise l'environnement pour l'éditeur."""
        # Note: Cette fonction n'est plus appelée dans __init__ car l'initialisation a lieu dans agent.__init__
        pass

    def on_grid_click(self, event):
        """Gère les clics de la souris pour éditer la grille."""
        if self.training_in_progress or self.training_finished:
            return
            
        view_a = self.current_altitude_level.get()
        c = event.x // self.CELL_SIZE
        r = event.y // self.CELL_SIZE
        
        if not (0 <= c < self.agent.TAILLE_GRILLE_XY and 0 <= r < self.agent.TAILLE_GRILLE_XY):
            return
            
        coords_3d = (r, c, view_a)
        mode = self.edit_mode.get()

        # Suppression préalable
        if coords_3d in self.agent.coords_obstacles:
            self.agent.coords_obstacles.remove(coords_3d)
        if coords_3d in self.agent.zones_vent:
            self.agent.zones_vent.remove(coords_3d)
            
        # Ajout/Modification selon le mode
        if mode == "obstacle":
            if coords_3d != self.agent.coords_depart and coords_3d != self.agent.coords_cible:
                self.agent.coords_obstacles.append(coords_3d)
        elif mode == "vent":
            if coords_3d != self.agent.coords_depart and coords_3d != self.agent.coords_cible:
                self.agent.zones_vent.append(coords_3d)
        elif mode == "depart":
            self.agent.coords_depart = coords_3d
        elif mode == "cible":
            self.agent.coords_cible = coords_3d
        elif mode == "effacer":
            pass 
            
        self.draw_grid()
        self.update_episode_advice() # Mise à jour du conseil si la complexité change

    def on_altitude_change(self, val):
        if self.replay_running:
            return 
        
        # En mode entraînement, on montre soit le chemin optimal (si terminé), soit la grille
        if not self.training_finished or self.training_in_progress:
            self.draw_grid(self.current_path) # current_path sera le chemin live
        else:
            path_to_draw = getattr(self, 'final_optimal_path', None)
            self.draw_grid(path_to_draw)

    def draw_grid(self, path=None):
        self.canvas_grid.delete("all")
        cell_size = self.CELL_SIZE
        view_a = self.current_altitude_level.get() 
        
        # 1. Dessin de la grille de fond (Obstacles, Vent, Cible)
        for r in range(self.agent.TAILLE_GRILLE_XY):
            for c in range(self.agent.TAILLE_GRILLE_XY):
                x1, y1 = c * cell_size, r * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size
                
                coords_3d = (r, c, view_a)
                color = "lightgray" 
                content = ""
                
                if coords_3d in self.agent.zones_vent:
                    color = "lightblue"
                    content = "W" 
                if coords_3d in self.agent.coords_obstacles:
                    color = "black"
                    content = "X" 
                if coords_3d == self.agent.coords_cible:
                    color = "green"
                    content = "T" 

                self.canvas_grid.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")
                if content:
                     self.canvas_grid.create_text(x1 + cell_size/2, y1 + cell_size/2, text=content, fill="white" if color in ["black", "green"] else "black", font=('Arial', int(cell_size * 0.4), 'bold'))

        # 2. Dessin du Chemin (Replay ou Live)
        if path:
            for i, (r, c, a) in enumerate(path):
                if i > 0:
                    prev_r, prev_c, prev_a = path[i-1]
                    # Dessin de la ligne 2D
                    if a == view_a and prev_a == view_a:
                        x_center = c * cell_size + cell_size / 2
                        y_center = r * cell_size + cell_size / 2
                        prev_x = prev_c * cell_size + cell_size / 2
                        prev_y = prev_r * cell_size + cell_size / 2
                        self.canvas_grid.create_line(prev_x, prev_y, x_center, y_center, fill="blue", width=2)
                
                # Dessin du point du chemin
                if a == view_a:
                    x_center = c * cell_size + cell_size / 2
                    y_center = r * cell_size + cell_size / 2
                    
                    color = "blue"
                    if (r, c, a) == self.agent.coords_cible: color = "orange"
                    
                    self.canvas_grid.create_oval(x_center - 5, y_center - 5, x_center + 5, y_center + 5, fill=color, outline=color)

                    # Indication de mouvement 3D
                    if i > 0 and path[i-1][2] != view_a and (r, c, a) != self.agent.coords_depart:
                        prev_a = path[i-1][2]
                        text = "⬆" if prev_a < a else "⬇"
                        self.canvas_grid.create_text(x_center, y_center, text=text, fill="purple", font=('Arial', 14, 'bold'))

        # 3. Dessin de la position actuelle de l'Agent (point rouge)
        agent_coords = None
        if path and self.replay_running and self.current_step < len(path):
             agent_coords = path[self.current_step] 
        # En mode Live Training, le dernier point du chemin est l'agent
        elif self.training_in_progress and path:
             agent_coords = path[-1] 
        elif self.agent.coords_depart:
             agent_coords = self.agent.coords_depart 
        
        if agent_coords and agent_coords[2] == view_a:
            agent_r, agent_c, agent_a = agent_coords
            x_agent = agent_c * cell_size + cell_size / 2
            y_agent = agent_r * cell_size + cell_size / 2
            
            radius = cell_size * 0.3
            self.canvas_grid.create_oval(x_agent - radius, y_agent - radius, x_agent + radius, y_agent + radius, fill="red", outline="red")
            self.canvas_grid.create_text(x_agent, y_agent, text="A", fill="white", font=('Arial', int(cell_size * 0.4), 'bold'))

    def start_replay(self):
        if self.replay_running or not self.final_optimal_path:
            return
        
        self.current_path = self.final_optimal_path
        self.current_step = 0
        self.replay_running = True
        self.label_replay_status['text'] = "Statut: Replay en cours..."
        self.button_replay.config(text="Replay en cours...", state=tk.DISABLED)
        self.replay_step()

    def replay_step(self):
        if not self.master.winfo_exists(): return
        
        if self.current_step < len(self.current_path):
            current_coords = self.current_path[self.current_step]
            
            if self.current_altitude_level.get() != current_coords[2]:
                 self.current_altitude_level.set(current_coords[2])
            
            current_sub_path = self.current_path[:self.current_step + 1]
            self.draw_grid(current_sub_path)
            
            self.label_replay_status['text'] = f"Statut: Étape {self.current_step} / {len(self.current_path) - 1} (Alt: {current_coords[2]})"
            
            self.current_step += 1
            self.master.after(200, self.replay_step) 
        else:
            self.replay_running = False
            self.label_replay_status['text'] = "Statut: Replay terminé."
            self.button_replay.config(text="Rejouer Optimal", state=tk.NORMAL)
            self.draw_grid(self.final_optimal_path)

    def live_path_update(self, path):
        """Callback appelé par l'agent pour mettre à jour la visualisation en direct (NOUVEAU)."""
        if not self.master.winfo_exists() or not self.training_in_progress: return
        
        self.current_path = path # Garde le chemin de l'épisode en cours
        
        # Mise à jour de la grille 2D
        # Si on est dans l'onglet Éditeur/Replay, on dessine le chemin en direct
        if self.notebook.tab(self.notebook.select(), "text") == "Éditeur de Grille & Replay (2D)":
            # Change l'altitude pour suivre l'agent
            if path and path[-1][2] != self.current_altitude_level.get():
                self.current_altitude_level.set(path[-1][2])
            self.draw_grid(self.current_path)
        
        # Mise à jour du statut pour montrer l'altitude actuelle de l'agent
        if path:
             self.label_replay_status['text'] = f"Statut: Entraînement en cours (Alt: {path[-1][2]})"
        
    def update_training_status(self, episode, best_reward):
        if not self.master.winfo_exists(): return 
        self.progress_bar['value'] = episode
        self.label_episode['text'] = f"Episode: {episode} / {self.params.EPISODES}"
        if best_reward > -float('inf'):
            self.label_best_reward['text'] = f"Meilleure Récompense: {best_reward:.2f}"
        
    def training_complete(self):
        if not self.master.winfo_exists(): return
        
        self.training_in_progress = False
        self.training_finished = True
        
        messagebox.showinfo("Entrainement Terminé", f"Le Q-learning est terminé après {self.params.EPISODES} épisodes.\nMeilleure Récompense: {self.agent.meilleure_recompense:.2f}")
        
        Q_table = self.agent.Q_table
        self.final_optimal_path = obtenir_chemin_optimal(
            Q_table, self.agent.coords_depart, self.agent.coords_cible, 
            self.agent.coords_obstacles, self.agent.zones_vent,
            self.agent.TAILLE_GRILLE_XY, self.agent.NIVEAUX_ALTITUDE
        )
        
        self.current_altitude_level.set(self.agent.coords_depart[2]) 
        self.draw_grid(self.final_optimal_path)
        self.plot_convergence()
        self.plot_path_statistics()
        self.draw_3d_environment(self.final_optimal_path)
        
        self.button_start_train.config(text="Lancer l'entraînement", state=tk.NORMAL)
        self.button_replay.config(state=tk.NORMAL)
        self.altitude_slider.config(state=tk.NORMAL) 
        
        for child in self.editor_frame.winfo_children():
             child.config(state=tk.NORMAL)
             
        self.button_export_map.config(state=tk.NORMAL)
        self.button_import_map.config(state=tk.DISABLED) 
        
        if self.final_optimal_path and self.final_optimal_path[-1] == self.agent.coords_cible:
             self.label_replay_status['text'] = f"Statut: Prêt (Longueur: {len(self.final_optimal_path) - 1} étapes)"
        else:
             self.label_replay_status['text'] = "Statut: Échec (Cible non atteinte)"
             
    def start_training_manually(self):
        """Lance l'entraînement lorsque l'utilisateur clique sur le bouton."""
        
        if self.agent.coords_depart is None or self.agent.coords_cible is None:
            messagebox.showerror("Erreur", "Veuillez placer un point de Départ (A) et une Cible (T).")
            return
            
        if self.agent.coords_depart == self.agent.coords_cible:
             messagebox.showwarning("Avertissement", "Le Départ et la Cible sont au même endroit.")

        self.training_in_progress = True
        self.training_finished = False
        
        self.button_start_train.config(text="Entraînement en cours...", state=tk.DISABLED)
        self.altitude_slider.config(state=tk.DISABLED)
        for child in self.editor_frame.winfo_children():
             child.config(state=tk.DISABLED)
             
        self.button_export_map.config(state=tk.DISABLED)
        self.button_import_map.config(state=tk.DISABLED)
        
        self.label_replay_status.config(text="Statut: Entraînement en cours...")

        threading.Thread(target=self.start_training, daemon=True).start()

    def start_training(self):
        def progress_callback(episode, best_reward):
            if self.master.winfo_exists():
                self.master.after(0, self.update_training_status, episode, best_reward)
                
        def live_path_callback(path):
            if self.master.winfo_exists():
                self.master.after(0, self.live_path_update, path)
            
        # Lancement de l'entraînement avec le callback de chemin en direct
        Q_table, meilleur_chemin, meilleure_recompense, historique_recompenses = self.agent.train(
            progress_callback=progress_callback, 
            live_path_callback=live_path_callback
        )
        
        if self.master.winfo_exists():
            self.master.after(0, self.training_complete) 

    def plot_convergence(self):
        if not self.agent.historique_recompenses:
            self.ax_conv.clear()
            self.ax_conv.text(0.5, 0.5, "En attente des données d'entraînement...", ha='center', va='center', fontsize=12)
            self.ax_conv.set_title("Historique d'Apprentissage Q-Learning (Drone 3D)")
            self.canvas_chart.draw()
            return

        historique_recompenses = self.agent.historique_recompenses
        fenetre_lissage = 100
        
        if len(historique_recompenses) < fenetre_lissage:
             fenetre_lissage = max(1, len(historique_recompenses))

        recompenses_moyenne_glissante = np.convolve(historique_recompenses, np.ones(fenetre_lissage)/fenetre_lissage, mode='valid')
        
        self.ax_conv.clear()
        self.ax_conv.plot(historique_recompenses, alpha=0.3, color='gray', label="Récompense par Épisode (Brut)")
        
        if recompenses_moyenne_glissante.size > 0:
            self.ax_conv.plot(np.arange(len(recompenses_moyenne_glissante)) + fenetre_lissage - 1, 
                             recompenses_moyenne_glissante, 
                             color='red', 
                             label=f"Moyenne Glissante (Fenêtre {fenetre_lissage})")
        
        self.ax_conv.set_title("Convergence de l'apprentissage Q-Learning")
        self.ax_conv.set_xlabel("Numéro d'Épisode")
        self.ax_conv.set_ylabel("Récompense Cumulée")
        self.ax_conv.legend()
        self.ax_conv.grid(True, linestyle='--', alpha=0.6)
        
        self.canvas_chart.draw()

    def get_path_rewards(self, path, agent):
        rewards = []
        cum_r = 0
        
        if not path:
            return []
            
        for i in range(1, len(path)): 
            state = path[i]
            prev_state = path[i-1]
            r_step = 0
            
            if state == agent.coords_cible:
                r_step = RECOMPENSE_CIBLE
            else:
                if state[2] != prev_state[2]:
                    r_step = RECOMPENSE_MOUVEMENT_3D
                else: 
                    r_step = RECOMPENSE_MOUVEMENT_2D
                
                if state in agent.zones_vent:
                    r_step += RECOMPENSE_PENALITE_VENT
            
            cum_r += r_step
            rewards.append(cum_r)
            
        return rewards

    def plot_path_statistics(self):
        path = getattr(self, 'final_optimal_path', None)
        
        self.ax_pos.clear()
        self.ax_alt.clear()
        self.ax_reward.clear()
        
        if not path or len(path) <= 1:
            self.fig_stats.text(0.5, 0.5, "En attente du calcul du chemin optimal...", 
                                 ha='center', va='center', fontsize=12, color='gray')
            self.ax_pos.set_title("Position X/Y vs. Étape")
            self.ax_alt.set_title("Altitude (Z) vs. Étape")
            self.ax_reward.set_title("Récompense Cumulée vs. Étape")
            self.canvas_stats.draw()
            return

        steps = np.arange(len(path))
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        alts = [p[2] for p in path]
        
        # 1. Position X/Y Plot
        self.ax_pos.plot(steps, cols, label="Position X (Colonne)", color='blue')
        self.ax_pos.plot(steps, rows, label="Position Y (Ligne)", color='orange')
        self.ax_pos.scatter([0], [cols[0]], color='blue', marker='o', s=50)
        self.ax_pos.scatter([0], [rows[0]], color='orange', marker='o', s=50)
        self.ax_pos.scatter([steps[-1]], [cols[-1]], color='blue', marker='*', s=150)
        self.ax_pos.scatter([steps[-1]], [rows[-1]], color='orange', marker='*', s=150)
        self.ax_pos.set_ylabel("Coordonnée de Grille")
        self.ax_pos.legend()
        self.ax_pos.grid(True, linestyle='--', alpha=0.6)
        self.ax_pos.set_title("Position X/Y vs. Étape du Chemin Optimal")
        
        # 2. Altitude (Z) Plot
        self.ax_alt.plot(steps, alts, label="Altitude Z", color='green', marker='.')
        self.ax_alt.scatter([0], [alts[0]], color='green', marker='o', s=50)
        self.ax_alt.scatter([steps[-1]], [alts[-1]], color='green', marker='*', s=150)
        self.ax_alt.set_yticks(range(self.agent.NIVEAUX_ALTITUDE))
        self.ax_alt.set_ylabel("Niveau d'Altitude (A)")
        self.ax_alt.legend()
        self.ax_alt.grid(True, linestyle='--', alpha=0.6)
        self.ax_alt.set_title("Altitude (Z) vs. Étape du Chemin Optimal")

        # 3. Cumulative Reward Plot
        rewards = self.get_path_rewards(path, self.agent)
        if rewards:
            reward_steps = np.arange(1, len(path))
            self.ax_reward.plot(reward_steps, rewards, label="Récompense Cumulée", color='red')
            self.ax_reward.scatter([reward_steps[-1]], [rewards[-1]], color='red', marker='*', s=150)
            self.ax_reward.set_xlabel("Étape")
            self.ax_reward.set_ylabel("Récompense Cumulée")
            self.ax_reward.legend()
            self.ax_reward.grid(True, linestyle='--', alpha=0.6)
            self.ax_reward.set_title("Récompense Cumulée vs. Étape du Chemin Optimal")
        
        self.fig_stats.tight_layout()
        self.canvas_stats.draw()

# --- 5. Lancement de l'Application (MODIFIÉ) ---
if __name__ == "__main__":
    
    params = QLearningParameters(TAILLE_ETAT_BASE, NB_ACTIONS)
    agent = QLearningAgent(params, TAILLE_GRILLE_XY, NIVEAUX_ALTITUDE)
    
    root = tk.Tk()
    gui = QLearningGUI(root, agent, params)
    root.mainloop()
