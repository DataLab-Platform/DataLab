# DataLab — Plan d'implémentation des plugins métier et évolution du système de plugins

## 1. Objectif

L'objectif est de faire évoluer DataLab d'une plateforme scientifique généraliste, dont l'utilisateur doit d'abord comprendre les possibilités, vers une plateforme proposant des **applications métier immédiatement identifiables**.

Les trois premiers cas d'usage retenus sont :

1. **Camera & Detector Characterization**
2. **Pulse & Transient Characterization**
3. **Radiographic NDT Analysis**

Ils ont été choisis parce qu'ils :

- correspondent directement au modèle actuel de DataLab : signaux 1D et images 2D ;
- peuvent exploiter une part importante du catalogue Sigima existant ;
- permettent de produire des démonstrations visuelles et immédiatement compréhensibles ;
- correspondent à des besoins scientifiques ou industriels récurrents ;
- permettent de mettre en avant l'indépendance vis-à-vis des fabricants d'instruments ;
- exploitent les points différenciants de DataLab : traitements validés, reproductibilité, automatisation, Desktop/Web/Notebook et fonctionnement local ;
- constituent des terrains crédibles pour tester la capacité de DataLab à construire une communauté autour de plugins externes.

Le CND et le contrôle qualité figurent déjà parmi les domaines identifiés comme particulièrement pertinents pour DataLab, notamment en raison des capacités de traitement d'image, ROI, détection et du fonctionnement de DataLab-Web dans des environnements industriels contraints.

---

## 2. Principe architectural général

Les plugins ne doivent pas devenir trois applications indépendantes ni trois forks spécialisés de DataLab.

L'architecture cible est :

```text
                   DataLab Platform
                          │
                 DataLab Plugin SDK
                          │
          ┌───────────────┼───────────────┐
          │               │               │
       Camera           Pulse       Radiographic
        plugin          plugin         NDT plugin
          │               │               │
          └───────────────┼───────────────┘
                          │
               noyau et workflow headless
                          │
             ┌────────────┴────────────┐
             │                         │
          Sigima                 code scientifique
         générique                spécifique plugin
             │                         │
             └────────────┬────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
       adaptateur Desktop       adaptateur Web
```

La règle fondamentale est :

> **DataLab fournit l'environnement ; Sigima fournit les primitives scientifiques génériques ; le plugin fournit le vocabulaire, le protocole et le workflow métier.**

---

## 3. Séparation des responsabilités

### 3.1. Sigima

Une fonction doit rejoindre Sigima lorsqu'elle est scientifiquement générique et réutilisable indépendamment d'un plugin.

Exemples :

```text
estimate_time_delay()
detect_signal_saturation()
compute_linearity_error()
compute_contrast_to_noise_ratio()
measure_connected_components()
```

Une fonction ne doit pas rejoindre Sigima uniquement parce qu'un plugin en a besoin.

Par exemple :

```text
run_emva_characterization_campaign()
classify_radiographic_indication()
generate_ndt_inspection_summary()
```

sont des fonctions métier et doivent rester dans le plugin.

#### Règle pratique

Déplacer une fonction dans Sigima si :

- elle a un sens scientifique hors du workflow courant ;
- elle pourrait être utilisée par au moins deux domaines ;
- elle peut être documentée sans faire référence à l'application métier ;
- elle peut être testée sur des données synthétiques indépendantes.

Sinon, elle reste dans le plugin.

---

### 3.2. Package scientifique du plugin

Chaque plugin doit contenir un noyau utilisable sans DataLab GUI.

Organisation type :

```text
src/
└── datalab_camera_characterization/
    ├── algorithms/
    │   ├── temporal.py
    │   ├── spatial.py
    │   └── validation.py
    ├── processing.py
    ├── workflow.py
    ├── params.py
    └── plugin.py
```

Les trois premières couches doivent être sans dépendance Qt.

---

### 3.3. Workflow métier

C'est le composant le plus important du plugin.

Un utilisateur ne doit pas avoir à connaître :

```text
Compute > Image > Statistics
Compute > Image > Flat-field
Compute > Image > Projection
...
```

Il doit voir :

```text
Run camera characterization
```

Le workflow :

- identifie les entrées ;
- vérifie leur cohérence ;
- exécute les calculs ;
- produit les objets intermédiaires utiles ;
- agrège les résultats ;
- signale les données invalides ;
- présente une synthèse.

---

### 3.4. Interface DataLab

La couche DataLab doit rester mince :

- création d'actions ;
- sélection des entrées ;
- formulaires `guidata.DataSet` ;
- lancement du workflow ;
- affichage des résultats ;
- interactions utilisateur spécifiques.

Le calcul scientifique ne doit pas dépendre de cette couche.

---

## 4. Évolution du système de plugins DataLab

**Existant.** Le système Desktop fournit la découverte des modules `datalab_*`, l'accès aux panneaux et au proxy, la création différée des actions et le hot-reload. `DLMainWindow.reload_plugins()` désenregistre les instances, appelle `BaseActionHandler.clear_plugin_actions()` sur les deux panneaux, recharge les modules, recrée les plugins puis leurs actions. Le nettoyage des actions et sous-menus est donc déjà pris en charge.

Le défaut critique se situe dans les traitements externes : `BaseProcessor.computing_registry` associe une fonction à un `ComputingFeature`, sans identifiant stable, propriétaire ni API de suppression. Une fonction recréée après reload peut donc coexister avec l'ancienne et la recherche historique par nom n'est pas suffisamment déterministe.

**Cible.** L'objectif n'est pas de remplacer ce système mais de le transformer progressivement en SDK distribuable : identité stable, contributions possédées et retirables, recettes headless, exemples packagés, découverte Desktop par entry points et contrat de portabilité Web vérifié.

La base contient une intention de roadmap « Contributor Experience » et des templates de plugins utilisés par les tests. Elle ne contient pas encore de générateur de projet utilisable comme produit. Le générateur décrit plus loin est donc un livrable cible, pas une capacité existante à fusionner.

---

## 5. Plugin SDK — P0 : identité stable et métadonnées

**Existant.** `PluginInfo` contient uniquement `name`, `version`, `description` et `icon`. `PluginRegistry` recherche et déduplique les plugins par nom affiché ; `Conf.main.plugins_enabled_list` persiste également ces noms.

**Cible.** Chaque plugin doit disposer d'un identifiant technique namespacé indépendant de son nom affiché.

Exemple :

```python
PluginInfo(
    id="org.datalab.camera-characterization",
    name="Camera & Detector Characterization",
    version="0.1.0",
)
```

Propriétés runtime minimales proposées :

```text
id
name
version
description
icon
minimum_datalab_version
capabilities
object_types
web_status
```

`capabilities` utilise uniquement les valeurs consommées par DataLab : `PROCESSING`, `IO`, `VISUALIZATION` et `APPLICATION`. `object_types` indique les objets supportés (`SIGNAL`, `IMAGE`). Auteur, licence, site et dépôt restent dans les métadonnées du package Python tant qu'aucun comportement runtime ne les consomme.

`web_status` n'est pas un booléen auto-déclaré. Il vaut `unsupported`, `untested` ou `verified` et est associé à la matrice de versions réellement testée. Seul un test DataLab-Web réussi peut établir `verified`.

L'activation des plugins doit être enregistrée par `id`, et non par nom humain.

Cela permet de renommer ultérieurement :

```text
Radiographic NDT Analysis
```

en :

```text
Digital Radiography Inspection
```

sans perdre la configuration utilisateur.

La migration doit convertir l'ancienne liste de noms vers les IDs lorsque la correspondance est unique, conserver l'ancienne option en lecture pendant la transition et signaler toute ambiguïté. Un plugin historique sans ID reçoit temporairement un identifiant legacy déterministe avec avertissement ; les nouveaux plugins doivent fournir leur ID.

---

## 6. Plugin SDK — P0 : cycle de vie des traitements externes

C'est l'évolution technique la plus importante.

**Existant.** `PluginBase.register()` enregistre l'instance puis appelle `register_hooks()`. Une fois les panneaux initialisés, `DLMainWindow.__create_plugins_actions()` appelle séparément `create_actions()`. Au retrait, `PluginBase.unregister()` retire l'instance du registre puis appelle `unregister_hooks()`. Lors d'un reload global, la fenêtre nettoie ensuite les actions des deux panneaux.

Ce cycle est déjà suffisant pour les actions, mais pas pour les `ComputingFeature`, qui ne connaissent ni leur propriétaire ni une opération de suppression.

**Cible.** Un plugin métier doit pouvoir enregistrer de véritables fonctions de calcul dans les processeurs DataLab, puis les retirer proprement. Chaque contribution reçoit un `feature_id` namespacé et un `owner_plugin_id`.

Le registre des fonctionnalités du `Processor` doit donc devenir symétrique.

Conceptuellement :

```python
processor.add_feature(feature, owner=plugin_id)

processor.remove_feature(feature_id)

processor.remove_features_by_owner(plugin_id)
```

Le cycle de vie devient :

```text
plugin load
    ↓
register instance
    ↓
register hooks et computations possédées
    ↓
create actions après initialisation des panneaux
```

et :

```text
plugin unload / reload
    ↓
unregister hooks
    ↓
remove computations by owner
    ↓
remove instance
        ↓
clear/rebuild plugin actions lors d'un reload global
```

L'implémentation exacte peut conserver l'ordre interne actuel si le retrait reste transactionnel et laisse le registre cohérent en cas d'exception. Les anciens appels par fonction ou nom restent des alias de transition uniquement lorsqu'ils sont non ambigus. Toute collision de `feature_id` doit échouer explicitement.

### Critère essentiel

Ce scénario doit être garanti par test :

```text
load plugin
→ register processing
→ execute it
→ reload plugin
→ processing still exists exactly once
→ execute again
→ unload plugin
→ processing no longer exists
```

Cette mécanique est indispensable au hot-reload et au développement confortable des plugins métier.

---

## 7. Plugin SDK — P0 : distinguer les plugins « application »

Il ne faut pas créer un deuxième système de plugins.

**Cible.** Il faut pouvoir distinguer un plugin technique d'un plugin constituant un point d'entrée utilisateur. Cette notion n'existe pas encore dans `PluginInfo`.

Exemples de capacités :

```text
PROCESSING
IO
VISUALIZATION
APPLICATION
```

Un plugin pourrait en déclarer plusieurs.

Les trois plugins de cette feuille de route auront au minimum :

```text
APPLICATION + PROCESSING
```

et éventuellement :

```text
IO
```

pour la gestion de formats métier.

N'ajouter une nouvelle capacité que lorsqu'un consommateur DataLab lui donne un comportement précis dans l'UI, le cycle de vie ou la validation.

---

## 8. Plugin SDK — P0 : recettes métier

Il ne faut pas développer immédiatement un moteur graphique de workflow.

Une abstraction headless explicite suffit, mais elle doit être plus précise qu'un simple callable non typé.

Exemple conceptuel :

```python
@dataclass(frozen=True)
class RecipeInputSlot:
        id: str
        object_type: Literal["signal", "image"]
        cardinality: Literal["one", "many"]
        required: bool = True

@dataclass(frozen=True)
class RecipeDescriptor:
        recipe_id: str
    title: str
    description: str
    version: str
        inputs: tuple[RecipeInputSlot, ...]
    parameter_class: type[DataSet] | None
        run: Callable[..., RecipeOutcome]

@dataclass
class RecipeOutcome:
        objects: list[SignalObj | ImageObj]
        results: list[TableResult | GeometryResult]
        anchor_ids: dict[str, str]
        diagnostics: list[RecipeDiagnostic]
```

L'identifiant complet d'une recette est namespacé par l'ID du plugin. Les diagnostics ont un niveau, un code stable et un message. Le contrat prévoit progression et annulation, sans imposer un thread ou une technologie d'interface.

Chaque plugin expose plusieurs recettes.

### Camera

```text
Quick Camera Check
Temporal Noise Characterization
Spatial Uniformity Characterization
Complete Detector Characterization
```

### Pulse

```text
Single Pulse Characterization
Shot-to-Shot Analysis
Multi-channel Timing Analysis
```

### Radiographic NDT

```text
Radiograph Quality Check
Porosity Inspection
Indication Measurement
```

Les recettes sont :

- versionnées ;
- paramétrables ;
- appelables depuis l'interface ;
- appelables depuis Python ;
- testables sans interface ;
- traçables localement sur les objets qu'elles produisent.

Un `RecipeRunner` transactionnel doit résoudre et valider les slots, exécuter le callable avant toute mutation du workspace, puis committer groupes et objets uniquement en cas de succès. Comme `TableResult` et `GeometryResult` sont stockés dans les métadonnées d'un `SignalObj` ou `ImageObj`, le résultat doit désigner explicitement l'objet ancre auquel chaque table ou géométrie est attachée.

---

## 9. Plugin SDK — P0 : exemples exécutables

**Cible.** Un plugin application doit pouvoir déclarer des exemples packagés. Aucune API générique `PluginExample` n'existe actuellement.

Conceptuellement :

```python
PluginExample(
    id="camera-quickstart",
    title="Scientific CMOS quick characterization",
    description="...",
        resource="datalab_camera_characterization:examples/scmos.h5",
        recipe_id="org.datalab.camera-characterization:quick-camera-check",
        expected_checks=(...),
)
```

La ressource est résolue avec `importlib.resources` ; elle ne dépend ni du répertoire courant ni d'un chemin de développement.

Chaque plugin doit fournir au minimum :

```text
quickstart
realistic example
```

L'objectif n'est pas seulement documentaire.

L'utilisateur doit pouvoir cliquer sur :

```text
Open example
```

et obtenir immédiatement :

```text
données
+ paramètres
+ résultats attendus
+ recette disponible
```

---

## 10. Plugin SDK — P0/P1 : découverte par Python entry points

**Existant Desktop.** `discover_plugins()` parcourt les modules importables dont le nom commence par `datalab_`. Cette convention reste utile pour le développement local, les plugins intégrés et le standalone.

**Cible Desktop/pip.** Les plugins installés comme packages Python doivent aussi pouvoir utiliser les entry points standards :

```toml
[project.entry-points."datalab.plugins"]
camera-characterization =
    "datalab_camera_characterization.plugin:CameraPlugin"
```

La découverte combine alors :

```text
plugins installés déclarés par entry point
plugins locaux ou intégrés découverts par convention
```

Ces sources n'ont pas de priorité implicite. Deux contributions portant le même ID stable produisent un diagnostic de collision et aucune n'est choisie silencieusement. La source de découverte est conservée pour le diagnostic.

L'installation devient naturellement :

```bash
pip install datalab-camera-characterization
```

Ce mécanisme doit être mis en place avant de réfléchir à une marketplace.

Il ne constitue pas un mécanisme de distribution Web. DataLab-Web charge des plugins intégrés au bundle ou des sources explicites dans Pyodide ; un entry point Python installé sur le poste hôte n'y est pas visible. La garantie entry point vise l'installation Python/pip Desktop, pas automatiquement les plugins externes d'un exécutable figé.

---

## 11. Générateur officiel de plugins

**État actuel.** Le dépôt DataLab contient des exemples et templates de tests, ainsi qu'une intention de roadmap Contributor Experience. Il n'existe pas encore de générateur officiel installé et maintenu.

**Cible.** Une commande dédiée évite d'entrer en conflit avec le lanceur GUI `datalab` existant :

```bash
datalab-plugin create
```

Le générateur doit proposer notamment :

```text
Plugin name
Package name
Plugin ID
Description
License
Application / Processing / IO / Visualization
Signals?
Images?
DataSet parameters?
Example workspace?
PyPI publication?
Web target (`unsupported` ou `untested` à la création)?
```

Il génère :

```text
pyproject.toml
package structure
plugin descriptor
sample processing
sample recipe
sample DataSet
tests
documentation
GitHub Actions
Ruff configuration
typing
README
CONTRIBUTING
LICENSE
```

La première version est un squelette mince construit après stabilisation des descripteurs du SDK. Le dépôt Camera est créé avec cette version et sert de test grandeur nature. Le template n'est durci qu'à partir des difficultés réellement rencontrées par Camera ; Pulse puis NDT utilisent ensuite la version éprouvée. Il n'est donc pas nécessaire de générer les trois dépôts dès le démarrage.

---

## 12. Ne pas introduire un nouveau modèle d'objet « Campaign »

Les trois plugins ont besoin de la notion de campagne.

Mais il serait prématuré de créer :

```text
CameraCampaignObj
PulseCampaignObj
NdtInspectionObj
```

Le modèle `SignalObj` / `ImageObj` / groupes / résultats doit rester la base.

**Existant.** Chaque panneau Signal ou Image possède son propre `ObjectModel`. Un `ObjectGroup` contient un UUID, un titre et une liste d'UUID d'objets ; il ne contient ni groupe enfant ni métadonnées. Les groupes et UUID sont persistés dans le workspace HDF5. Une même campagne qui produit signaux et images ne peut donc pas être contenue dans un groupe unique.

L'organisation doit plutôt reposer sur :

- groupes plats, utilisés pour l'organisation visuelle ;
- métadonnées namespacées ;
- UUID et paramètres de traitement déjà persistés par DataLab ;
- recettes ;
- objets dérivés et résultats attachés à un objet ancre.

Exemple :

```text
Panneau Image
├── Camera / Dark
├── Camera / Flat / Exposure 1
├── Camera / Flat / Exposure 2
└── Camera / Spatial results

Panneau Signal
└── Camera / Temporal results
```

Les barres obliques font partie du titre des groupes ; elles ne représentent pas une hiérarchie. Les objets produits dans les deux panneaux sont reliés par le même `run_id` dans leurs métadonnées.

---

## 13. Métadonnées métier

Utiliser des clés namespacées par l'ID stable du plugin, sous la forme `plugin.<plugin-id>.<key>`. Les valeurs doivent être sérialisables en HDF5 et en JSON ; aucun objet Python métier opaque ne doit être stocké dans les métadonnées.

Exemples caméra :

```text
plugin.org.datalab.camera-characterization.role = dark
plugin.org.datalab.camera-characterization.series = exposure_003
plugin.org.datalab.camera-characterization.exposure_time = ...
plugin.org.datalab.camera-characterization.gain_mode = ...
```

Pulse :

```text
plugin.org.datalab.pulse-characterization.shot = 173
plugin.org.datalab.pulse-characterization.channel = CH1
plugin.org.datalab.pulse-characterization.trigger = ...
```

CND :

```text
plugin.org.datalab.radiographic-ndt.modality = digital_radiography
plugin.org.datalab.radiographic-ndt.role = source
plugin.org.datalab.radiographic-ndt.inspection_id = ...
plugin.org.datalab.radiographic-ndt.material = ...
```

Ces métadonnées doivent rester sérialisables en HDF5 et en JSON. Elles sont portées par les `SignalObj` et `ImageObj`, car `ObjectGroup` ne possède actuellement ni dictionnaire de métadonnées ni groupes enfants.

---

## 14. Trace locale d'exécution des recettes

**Existant.** `ProcessingParameters` enregistre déjà, dans les métadonnées des objets dérivés, le nom et le pattern d'un traitement, ses paramètres et les UUID de ses sources. Les UUID des objets et groupes sont préservés lors du rechargement d'un workspace HDF5. Ce mécanisme décrit une transformation unitaire ; il ne constitue pas un historique global de workflow.

**Cible SDK.** Chaque recette ajoute un `RecipeRunRecord` versionné à tous les objets qu'elle produit. Cette trace locale doit contenir au minimum :

```text
run ID
plugin ID et version
recipe ID et version
paramètres résolus sérialisés en JSON
UUID des objets d'entrée et de sortie
versions DataLab et Sigima
statut et horodatages
```

Le même `run ID` relie les sorties réparties entre les panneaux Signal et Image. Cette relation est portée par les objets, pas par un groupe commun : DataLab maintient un modèle de groupes distinct dans chaque panneau.

Ce choix permet de répondre localement à :

> Comment ce résultat a-t-il été obtenu ?

sans introduire de DAG global, de manifeste de workspace ni de nouveau modèle d'objet. Les `TableResult` et `GeometryResult` restant stockés dans les métadonnées d'un objet, une recette doit désigner un objet de sortie ancre pour chaque résultat consolidé.

---

## 15. Expérience utilisateur « Applications »

Une fois deux plugins crédibles disponibles, DataLab doit pouvoir les présenter dans une vue plus visible que le simple menu Plugins.

Par exemple :

```text
Applications
──────────────────────────────────────────

Camera & Detector Characterization

Qualify camera noise, response and
spatial uniformity.

[Start analysis]  [Open example]


Pulse & Transient Characterization

Analyze repeated pulse acquisitions,
timing and shot-to-shot stability.

[Start analysis]  [Open example]


Radiographic NDT Analysis

Inspect and quantify indications in
digital radiographs.

[Start analysis]  [Open example]
```

Ce n'est pas une nouvelle application.

C'est une **nouvelle porte d'entrée dans DataLab**.

---

## 16. Plugin 1 — Camera & Detector Characterization

### 16.1. Positionnement

Promesse cible :

> **Characterize and compare scientific cameras and detectors independently of their manufacturer.**

Le plugin ne doit pas concurrencer le logiciel d'acquisition du constructeur.

Le workflow commence après l'acquisition.

**État actuel.** Sigima fournit plusieurs primitives nécessaires, mais DataLab ne contient ni importeur de campagne Camera, ni métriques de caractérisation dédiées, ni simulateur Camera, ni workflow consolidé. Le premier pilote doit donc être présenté comme un prototype de caractérisation relative, pas comme une capacité déjà disponible.

---

## 17. Deux niveaux de caractérisation caméra

Il est important de séparer deux niveaux.

### Niveau A — caractérisation sans source photométrique étalonnée

Accessible à pratiquement tout utilisateur.

À partir de séries dark et flat :

- bruit temporel ;
- variance ;
- linéarité relative ;
- saturation ;
- uniformité ;
- cartes DSNU-like et PRNU-like relatives, avec protocole et normalisation explicités ;
- pixels aberrants ;
- comparaison des réglages.

Ce niveau, exprimé en DN et sans revendication métrologique, constitue le MVP. Les termes DSNU/PRNU ne doivent être employés sans suffixe ou réserve qu'après définition et validation d'un protocole compatible avec leur interprétation métrologique.

### Niveau B — caractérisation métrologique

Avec connaissance de l'éclairement ou du flux photonique :

- responsivité ;
- rendement quantique si les conditions le permettent ;
- gain de conversion ;
- sensibilité ;
- mesures plus proches d'un protocole EMVA 1288.

Cette partie doit arriver après validation scientifique du MVP.

Le plugin ne doit pas utiliser le terme « EMVA 1288 compliant » tant qu'une conformité rigoureuse et documentée n'a pas été établie.

Le gain de conversion, le rendement quantique et les grandeurs dépendant d'un flux photonique connu sont exclus du MVP relatif.

---

## 18. Workflow Camera

**Cible.** Aucun des enchaînements campagne ci-dessous n'existe actuellement comme API ou assistant générique :

```text
New Camera Characterization
        ↓
Describe camera / acquisition
        ↓
Import dark series
        ↓
Import uniform illumination series
        ↓
Assign or infer acquisition roles
        ↓
Validate dataset
        ↓
Temporal analysis
        ↓
Spatial analysis
        ↓
Generate consolidated results
        ↓
Save workspace / export tabular results
```

L'assignation automatique doit toujours produire des diagnostics vérifiables et permettre une correction utilisateur. Un moteur de rapport n'est pas requis pour le MVP.

---

## 19. Paramètres Camera

Un `CameraCharacterizationParam` pourrait comprendre :

```text
Manufacturer
Model
Serial number
Sensor technology
Pixel pitch
Bit depth
Gain mode
Binning
Sensor temperature
Wavelength
Exposure unit
Illumination calibration available
```

La plupart doivent rester optionnels.

Le plugin doit fonctionner avec des données imparfaitement documentées.

---

## 20. Validation des entrées Camera

Avant tout calcul :

- tailles d'images cohérentes ;
- type numérique cohérent ;
- présence d'images dark ;
- nombre minimal d'acquisitions ;
- saturation excessive ;
- variance anormale ;
- incohérences de métadonnées ;
- ordre des niveaux d'exposition.

Les anomalies doivent être affichées avant l'analyse.

Cette validation appartient au workflow Camera cible. Elle doit terminer avant toute mutation du workspace et retourner des diagnostics structurés plutôt que des erreurs tardives au milieu des calculs.

---

## 21. Calculs temporels Camera

**Existant et réutilisable.** Sigima fournit notamment l'agrégation moyenne/écart-type d'images, les statistiques d'image, les histogrammes et les profils. `sigima.proc.image.average()` accumule les images, tandis que `standard_deviation()` construit actuellement un tableau contenant toute la pile avant `numpy.std`.

**Cible plugin Camera.** Le workflow de campagne compose ces primitives et ajoute les métriques Camera manquantes :

MVP :

```text
mean signal vs exposure
variance vs signal
response curve
linearity residuals
temporal noise
saturation level
dynamic range relative estimate
SNR relative vs signal
```

Les courbes sont produites sous forme de `SignalObj`. Les métriques consolidées sont un `TableResult` attaché aux métadonnées de la courbe réponse choisie comme objet ancre ; elles ne deviennent pas des objets top-level du workspace.

Avant toute promesse de taille de campagne sur Desktop ou Web, l'écart-type doit être remplacé dans ce workflow par un calcul incrémental ou par blocs, validé numériquement contre NumPy et mesuré sur des piles représentatives, notamment en 2048².

---

## 22. Calculs spatiaux Camera

**Cible plugin Camera.** Les primitives de moyenne, profils et histogrammes existent ; les cartes et règles de caractérisation ci-dessous restent à implémenter et à valider :

MVP :

```text
mean dark image
mean illuminated image
dark non-uniformity
flat-field non-uniformity
relative DSNU-like map
relative PRNU-like map
bad-pixel candidates
row profile
column profile
histograms
```

Les cartes restent des `ImageObj`.

---

## 23. Résultat synthétique Camera

Le tableau est un `TableResult` attaché à l'objet ancre de la recette :

Exemple :

| Metric | Value | Unit | Status |
| --- | ---: | --- | --- |
| Temporal noise | … | DN | OK |
| Linearity error | … | % | OK |
| Saturation | … | DN | — |
| Spatial non-uniformity | … | % | Warning |
| Defective pixels | … | ppm | OK |

Le statut doit initialement provenir de seuils explicitement paramétrés, pas de règles prétendument normatives implicites. Les unités et hypothèses de chaque métrique sont enregistrées avec le résultat.

---

## 24. Simulateur Camera

**Cible plugin Camera.** Aucun simulateur Camera n'existe actuellement dans Sigima ou DataLab. Développer un générateur déterministe dans le noyau scientifique du plugin.

Paramètres :

```text
offset
conversion gain
read noise
shot noise
dark current
PRNU
DSNU
saturation
bit depth
defective pixels
```

Le générateur permet :

1. de produire des images synthétiques ;
2. de connaître exactement les paramètres réels ;
3. d'exécuter le plugin ;
4. de vérifier que les paramètres estimés sont dans les tolérances attendues.

C'est la base du mode de validation scientifique.

Le simulateur conserve sa graine et la vérité terrain dans un format testable. Le fait de simuler un gain de conversion ou un flux ne signifie pas que le MVP est autorisé à les estimer ou à revendiquer une conformité métrologique.

---

## 25. Phases Camera

### Camera C0 — moteur minimal

- simulateur ;
- séries dark/flat ;
- moyenne et variance incrémentales ou par blocs ;
- courbe réponse ;
- linéarité ;
- bruit.

### Camera C1 — caractérisation spatiale

- DSNU/PRNU ;
- pixels aberrants ;
- profils ;
- distributions.

### Camera C2 — workflow application

- assistant d'import ;
- validation ;
- recettes ;
- résultats consolidés ;
- exemple.

Ce jalon livre d'abord le workflow headless puis un adaptateur Desktop mince. En l'absence de données réelles documentées et d'une revue scientifique, il reste qualifié d'alpha non métrologique.

### Camera C3 — métrologie avancée

- flux photonique ;
- paramètres EMVA pertinents ;
- validation scientifique renforcée.

### Camera C4 — comparaison

```text
Camera A / setting A
Camera A / setting B
Camera B
```

avec tableaux comparatifs.

---

## 26. Critères d'acceptation Camera

Le quickstart doit permettre de :

```text
ouvrir l'exemple
→ lancer une caractérisation
→ obtenir les courbes temporelles
→ obtenir les cartes spatiales
→ obtenir le tableau synthétique
```

sans écrire de Python.

Les tests synthétiques doivent vérifier les paramètres estimés sur des jeux dont les caractéristiques sont connues.

Ils doivent aussi vérifier le déterminisme, les entrées invalides, l'équivalence numérique batch/incrémental, le budget mémoire, le round-trip HDF5 des UUID et du `RecipeRunRecord`, et l'attachement du tableau au bon objet ancre. Une release stable exige en plus des données réelles documentées et une revue scientifique indépendante du code.

---

## 27. Plugin 2 — Pulse & Transient Characterization

### 27.1. Positionnement

Promesse :

> **Turn repeated oscilloscope or digitizer acquisitions into a reproducible experimental campaign analysis.**

Le plugin ne remplace pas l'oscilloscope.

Il prend en charge ce qui arrive typiquement après :

```text
Save waveforms
→ Python / MATLAB / Excel
```

et remplace cette dernière étape.

**État actuel.** Sigima sait déjà extraire les caractéristiques d'une impulsion individuelle. Il ne fournit pas de modèle de campagne, de batch consolidé, de flags qualité métier, d'alignement de séries ni de simulateur de 500 tirs.

---

## 28. Capitaliser sur Sigima

**Existant.** `sigima.tools.signal.pulse.PulseFeatures` et `extract_pulse_features()` fournissent notamment : forme, polarité, amplitude, offset, durée de pied, plages de baseline/plateau, temps de montée et de descente, FWHM, puis positions `x0`, `x50` et `x100`. `sigima.proc.signal.extract_pulse_features()` expose ces mesures comme `TableResult` pour un `SignalObj`.

Le plugin doit réutiliser ce socle ainsi que les traitements génériques disponibles :

- pics ;
- correction d'offset ;
- filtrage ;
- statistiques ;
- FFT si nécessaire.

**Manquant pour le plugin.** L'intégrale et le SNR consolidés ne font pas partie de `PulseFeatures`. L'effort principal porte donc sur :

- traitement par campagne ;
- intégrale et SNR avec conventions explicites ;
- validation ;
- alignement ;
- consolidation ;
- classification qualité.

---

## 29. Workflow Pulse

**Cible plugin.** Ce workflow n'existe pas encore comme orchestration DataLab :

```text
New Pulse Campaign
       ↓
Import acquisitions
       ↓
Assign channels / shots
       ↓
Configure baseline and detection
       ↓
Preview representative pulses
       ↓
Validate parameters
       ↓
Run complete campaign
       ↓
Flag invalid acquisitions
       ↓
Generate shot-to-shot results
       ↓
Compare / export
```

---

## 30. MVP Pulse : une voie

Pour chaque impulsion, réutiliser les champs existants lorsqu'ils sont applicables :

```text
offset de baseline
amplitude
polarity
x0 / x50 / x100
rise time
fall time
FWHM
```

Ajouter dans le plugin, avec définitions et diagnostics explicites :

```text
integral
SNR
saturation flag
```

Résultats agrégés :

```text
amplitude vs shot
FWHM vs shot
integral vs shot
timing vs shot
histograms
mean pulse
standard deviation pulse
```

La table par tir et les courbes de synthèse sont produites par la recette de campagne. Un `SignalObj` de synthèse sert d'objet ancre à la table consolidée.

---

## 31. Qualité des acquisitions Pulse

**Cible plugin.** Chaque acquisition reçoit un état explicable :

```text
VALID
NO_PULSE
LOW_SNR
SATURATED
MULTIPLE_PULSES
OUTLIER
```

Le diagnostic doit être explicable.

Exemple :

```text
Shot 173
Status: SATURATED
Reason: 27 samples at ADC maximum
```

Pas de classification opaque dans le MVP.

Les seuils, la raison et les valeurs ayant déclenché le statut sont conservés dans la ligne de résultat. Aucun de ces statuts n'est actuellement fourni par `PulseFeatures`.

---

## 32. Alignement des impulsions

**Cible plugin.** Ajouter un workflow d'alignement headless avant agrégation.

Méthodes possibles :

```text
threshold crossing
peak
50 % crossing
cross-correlation
```

L'utilisateur doit pouvoir comparer :

```text
raw campaign
aligned campaign
```

La fonction d'alignement doit être headless.

Commencer par l'alignement sur pic ou franchissement de seuil. La corrélation croisée n'est ajoutée qu'après mesure du coût et de la qualité sur le jeu cible. Les performances de 500 tirs doivent être mesurées sous CPython et dans le Pyodide principal ; les workers macro/notebook de DataLab-Web ne constituent pas un moteur générique de recettes parallèles.

---

## 33. Pulse V2 — plusieurs voies

Une campagne peut contenir :

```text
CH1
CH2
CH3
...
```

Le plugin ajoute :

```text
delay CH1 → CH2
jitter
cross-correlation delay
amplitude correlation
integral correlation
missing-channel detection
```

C'est cette version qui devient particulièrement utile pour des diagnostics expérimentaux complexes.

---

## 34. Pulse V3 — comparaison de configurations

Exemple :

```text
Configuration A
Configuration B
Configuration C
```

Comparaison :

| Metric | A | B | C |
| --- | ---: | ---: | ---: |
| Mean amplitude | … | … | … |
| RMS variation | … | … | … |
| Mean FWHM | … | … | … |
| Timing jitter | … | … | … |
| Invalid shots | … | … | … |

---

## 35. Générateur synthétique Pulse

**Cible plugin.** Aucun générateur de campagne Pulse complet n'existe actuellement. Produire des séries déterministes avec vérité terrain :

```text
Gaussian pulses
asymmetric pulses
baseline drift
amplitude drift
timing jitter
white noise
saturation
missing pulses
double pulses
```

Le dataset de démonstration doit contenir volontairement plusieurs anomalies.

Exemple de scénario :

```text
500 shots
11 invalid acquisitions
slow amplitude drift
timing jitter increasing after shot 300
```

Le plugin doit les rendre immédiatement visibles.

---

## 36. Critères d'acceptation Pulse

Sur le workspace exemple :

```text
Open
→ Run Shot-to-Shot Analysis
→ invalid shots automatically highlighted
→ consolidated metrics generated
→ representative signals plotted
```

Le résultat doit être compréhensible sans avoir à inspecter 500 courbes individuellement.

Les tests vérifient les mesures et chaque flag sur la vérité terrain, le déterminisme, l'attachement de la table à l'objet ancre et le round-trip HDF5. Le choix de l'alignement et le budget UX sont fixés à partir des benchmarks CPython/Pyodide, sans objectif de performance inventé à l'avance. Une release stable exige aussi un jeu réel documenté et une revue scientifique des conventions d'intégrale, SNR et qualité.

---

## 37. Plugin 3 — Radiographic NDT Analysis

### 37.1. Périmètre

Le plugin ne doit pas essayer de couvrir « le CND ».

Il doit cibler :

> **2D digital radiography inspection and quantitative analysis.**

À exclure du MVP :

- tomographie 3D ;
- phased-array ultrasonics ;
- TFM/FMC ;
- thermographie temporelle ;
- contrôle automatisé de conformité à toutes les normes CND.

Ces sujets imposeraient d'autres modèles de données ou un effort métier disproportionné.

**État actuel.** Sigima fournit des briques génériques de seuillage, morphologie, détection de pics/contours/blobs, ROI, statistiques et résultats géométriques/tabulaires. Il ne fournit ni pipeline de porosité validé, ni modèle de candidat revu, ni interface métier accept/reject/classify/merge/split, ni protocole CND. Le plugin produit ne peut donc commencer qu'après accès à un expert CND et à un corpus exploitable avec annotations.

---

## 38. Positionnement NDT

Promesse :

> **Inspect, quantify and document indications in digital radiographs independently of the acquisition equipment.**

Le plugin commence après acquisition.

Il ne pilote ni :

- générateur X ;
- détecteur ;
- dose ;
- séquence d'acquisition.

Il analyse des images déjà produites.

---

## 39. Positionnement sécurité et métier

Le plugin doit être présenté comme un outil d'**assistance à l'analyse**.

Pas comme :

> automatic defect certification

mais comme :

> assisted detection, measurement and documentation of radiographic indications.

La validation humaine reste centrale :

```text
Automatic candidate detection
            ↓
      Review candidates
        ↙          ↘
    Accept        Reject
      ↓
Measurement / annotation
      ↓
Inspection results
```

---

## 40. Import NDT

### V1

Limiter le premier prototype aux formats déjà naturellement manipulables par DataLab :

```text
TIFF
PNG
formats d'image standards sans codec métier supplémentaire
```

et préserver :

- résolution ;
- échelle physique lorsqu'elle est fournie ou déjà portée par l'objet ;
- métadonnées disponibles.

### Chantier I/O séparé : DICOM/DICONDE

**Existant.** `DICOMImageFormat.read_data()` appelle `imread_dicom()`, qui lit `PixelData` avec pydicom et retourne un `numpy.ndarray`. Le chemin standard de création ajoute surtout la source du fichier ; il ne mappe pas les tags DICOM vers les métadonnées `ImageObj`, ne préserve pas les champs inconnus et ne constitue pas un lecteur DICONDE.

Une vraie prise en charge DICOM/DICONDE doit être étudiée comme un chantier indépendant :

Cette fonction doit être isolée dans une couche IO :

```text
io/diconde.py
```

Elle doit :

- lire les métadonnées pertinentes ;
- convertir l'image vers `ImageObj` ;
- mapper photométrie, pixel spacing et unités ;
- conserver les informations originales ;
- ne pas perdre silencieusement les champs non interprétés.

L'étude doit aussi couvrir les codecs, les variantes de pixel data et la disponibilité de pydicom et de ses dépendances dans Pyodide. L'écriture DICONDE ne doit venir qu'après stabilisation de la lecture, validation du mapping avec un expert et clarification des informations à produire.

---

## 41. Workflow NDT

**Cible conditionnelle.** Ce workflow n'est pas une API existante et ne doit pas être présenté comme un produit planifié tant que les gates données/expert ne sont pas franchies :

```text
New Radiographic Inspection
        ↓
Import radiograph
        ↓
Check metadata and physical scale
        ↓
Image quality assessment
        ↓
Define inspection region
        ↓
Preprocessing
        ↓
Candidate indication detection
        ↓
Human review
        ↓
Measurement
        ↓
Consolidated inspection results
```

---

## 42. NDT V1 — qualité de l'image

Avant de rechercher les défauts, le plugin cible compose les statistiques/profils existants et ajoute les métriques dont la définition a été validée :

```text
dynamic range
saturation
noise estimate
SNR
CNR between selected regions
local contrast
profiles
uniformity
```

Le plugin doit pouvoir afficher :

```text
Image quality: acceptable for analysis
```

ou :

```text
Warning: significant saturation
```

sans prétendre à une conformité normative tant que le protocole n'est pas explicitement implémenté.

La phrase « acceptable for analysis » n'est autorisée que si ses critères sont documentés et validés avec l'expert ; sinon l'interface affiche uniquement les mesures et avertissements factuels.

---

## 43. NDT V1 — premier défaut ciblé : porosités / indications compactes

Il faut volontairement commencer par un cas simple et démonstratif.

Je choisirais :

> **détection et mesure d'indications compactes de type porosité.**

Pourquoi :

- bonne adéquation aux traitements d'image existants ;
- segmentation accessible ;
- mesures géométriques naturelles ;
- visualisation immédiate ;
- possibilité de générer des données synthétiques ;
- validation humaine facile.

Pipeline expérimental possible, à valider sur synthétique puis réel :

```text
ROI
↓
background normalization
↓
local contrast enhancement
↓
candidate segmentation
↓
morphological cleanup
↓
connected components
↓
filtering
↓
review
```

---

## 44. Mesures NDT

Pour chaque indication :

```text
identifier
centroid
bounding box
area
equivalent diameter
major/minor dimensions
orientation
mean contrast
maximum contrast
distance to reference feature if relevant
status
comment
```

`GeometryResult`, annotations PlotPy et ROI sont trois représentations distinctes. Le pipeline produit d'abord un `GeometryResult` et un `TableResult` attachés aux métadonnées de l'image ancre. L'adaptateur UI peut ensuite matérialiser les géométries comme annotations et, lorsque le cas le justifie, créer des ROI ; il doit conserver un identifiant de candidat stable entre ces représentations.

---

## 45. Revue humaine NDT

**Cible métier absente du socle actuel.** L'éditeur ROI générique ne fournit pas le workflow de revue attendu. Une interface spécifique doit permettre :

```text
Candidate #12

[Accept]
[Reject]
[Merge]
[Split / edit ROI]
```

et éventuellement :

```text
Classification:
- porosity
- inclusion
- linear indication
- unknown
```

La classification doit rester manuelle dans un premier temps.

Accept/reject et commentaire constituent le premier lot. Merge/split modifient la géométrie et les liens table/annotation ; ils forment un lot dédié, d'abord Desktop puis Web après validation du modèle.

---

## 46. Résultats NDT

Le tableau ci-dessous est un `TableResult` attaché à l'image d'inspection ancre, et non un objet top-level :

Tableau :

| ID | Type | X | Y | Size | Contrast | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| I001 | Porosity | … | … | … | … | Accepted |
| I002 | Unknown | … | … | … | … | Rejected |

Les indications validées restent visibles comme annotations sur l'image.

Le statut de revue, la classe manuelle et le commentaire doivent survivre au round-trip HDF5 sans être confondus avec le statut technique `OK/WARNING/ERROR/INVALID` d'une recette.

---

## 47. NDT V2 — indications linéaires

Après stabilisation du cas porosité :

- détection de structures linéaires ;
- fissures potentielles ;
- longueur ;
- largeur ;
- orientation ;
- continuité.

Il faudra être plus prudent sur les faux positifs et la validation.

---

## 48. NDT V2/V3 — qualité radiographique spécialisée

Évolutions possibles :

- reconnaissance d'IQI ;
- mesure de visibilité de fils ou trous ;
- qualité d'image selon protocole explicite ;
- comparaison avec une radiographie de référence ;
- détection d'évolution entre deux inspections.

Ces fonctionnalités ne doivent être implémentées qu'avec un expert CND et des données de validation adaptées.

Elles ne font pas partie de la roadmap engagée tant que ces ressources ne sont pas identifiées.

---

## 49. Générateur synthétique NDT

Pour un spike uniquement, créer des radiographies synthétiques avec :

```text
background thickness variation
noise
detector artifacts
compact indications
linear indications
different contrasts
different sizes
overlapping indications
```

Conserver la vérité terrain :

```text
position
geometry
contrast
class
```

Cela permet d'évaluer :

```text
detection rate
false positives
measurement error
```

sans dépendre uniquement de données industrielles confidentielles.

Le synthétique permet de tester géométrie et algorithmes, mais ne valide ni le réalisme radiographique ni une procédure d'inspection.

---

## 50. Dataset réel NDT

Le plugin ne doit cependant pas être validé seulement sur du synthétique.

Il faudra obtenir au minimum un petit corpus :

- public ;
- ou anonymisé ;
- ou spécialement acquis ;
- ou fourni par un partenaire avec droit de redistribution.

Le dataset réaliste est indispensable à la crédibilité du plugin.

Il est également un gate de développement produit : sans corpus exploitable et expert CND, le travail s'arrête au spike synthétique et aucune promesse de détection opérationnelle n'est publiée.

---

## 51. Critères d'acceptation NDT

Le quickstart doit permettre :

```text
open radiograph
→ run inspection
→ display candidate indications
→ accept/reject candidates
→ measure accepted indications
→ export consolidated results
```

Le tout sans script Python.

Avant ce quickstart produit, il faut définir avec l'expert les métriques de détection et de mesure, documenter les faux positifs/faux négatifs, valider les liens candidat-géométrie-table et rendre la revue humaine persistante. Aucun jugement automatique de conformité n'est autorisé et une validation synthétique seule ne franchit pas le gate.

---

## 52. Socle transversal aux trois plugins

Les trois plugins vont révéler les mêmes besoins.

Il faut volontairement les factoriser au niveau du SDK lorsque deux plugins en ont réellement besoin. En dehors des descripteurs minimaux de recette, d'exemple et de diagnostic, les éléments des sections 53 à 56 sont des cibles possibles et non des APIs actuelles.

---

## 53. Import de campagnes

Besoin commun Camera/Pulse :

```text
Import directory
```

avec :

- pattern de nommage ;
- extraction de métadonnées ;
- regroupement ;
- prévisualisation ;
- détection des fichiers incohérents.

Il serait possible de créer ultérieurement une API générique :

```python
CampaignImporter
```

mais uniquement après avoir implémenté Camera puis Pulse et identifié ce qui est réellement commun.

Ne pas abstraire prématurément.

**État actuel.** Aucun `CampaignImporter` générique n'existe. Camera commence avec une assignation de rôles minimale propre au plugin ; Pulse fournit ensuite le second cas réel permettant de décider s'il faut extraire une abstraction commune.

---

## 54. Résultats consolidés

Les trois plugins ont besoin de transformer :

```text
N objets
```

en :

```text
un tableau de résultats
+ plusieurs courbes de synthèse
+ objets dérivés
```

**Existant.** `TableResult` et `GeometryResult` sont sérialisés dans les métadonnées d'un objet Signal/Image via les adaptateurs DataLab. Ils ne sont pas des objets top-level et ne peuvent pas être ajoutés seuls à un groupe.

La cible commune minimale est donc le contrat d'objet ancre de `RecipeOutcome`. Une abstraction de rapport ou de collection de résultats n'est envisagée qu'après retour de Camera et Pulse.

---

## 55. Statut et validation

**Cible SDK.** Un concept très léger de diagnostic technique peut être partagé :

```text
OK
WARNING
ERROR
INVALID
```

et éventuellement :

```text
reason
```

Cela permet d'identifier :

- image saturée ;
- tir absent ;
- acquisition mal échantillonnée ;
- radiographie impropre au traitement.

Ce statut ne doit pas être confondu avec un jugement normatif ou réglementaire.

Il ne remplace pas les états métier spécifiques tels que les flags Pulse ou l'acceptation humaine NDT. Les codes de diagnostic sont stables ; leur texte d'interface reste traduisible.

---

## 56. Rapports

Je ne ferais pas du moteur de rapport une condition du MVP.

Les premières versions peuvent produire :

```text
workspace HDF5
CSV results
figures
```

Puis ajouter un rapport HTML.

Un futur mécanisme de figures scientifiques pourra être évalué lorsqu'une API concrète sera disponible ; le plan ne le traite pas comme une dépendance existante.

Le PDF ne doit venir qu'en aval du HTML ou d'un modèle de rapport commun.

---

## 57. DataLab-Web

L'objectif stratégique impose que les plugins puissent à terme fonctionner dans DataLab-Web.

**Existant.** La plateforme Web exécute Sigima dans Pyodide et fournit un shim Qt-free de l'API plugin Desktop. Ce shim n'est pas « le même système » : il charge des plugins intégrés au bundle ou des sources explicites, appelle `create_actions()` pendant `register()`, et stocke traitements/actions/générateurs dans des registres portant une `origin`. `clear_origin()` permet déjà leur retrait au unload.

Le sous-ensemble n'est pas complet : le processeur Web expose `register_1_to_1`, `register_2_to_1` et `register_n_to_1`, mais pas tous les patterns Desktop ; `action_for()` est un stub Desktop-only ; les dialogues utilisables dans le navigateur ont des variantes async ; l'I/O passe par les capacités du navigateur et non par un système de fichiers arbitraire.

Il faut donc imposer dès le départ :

> aucune dépendance Qt dans le moteur scientifique ou le workflow.

Le noyau scientifique et le workflow sont partagés. Les adaptateurs Desktop et Web restent distincts lorsque UI, async, I/O ou packaging l'exigent. Utiliser exactement la même classe `PluginBase` sur les deux plateformes n'est pas un objectif P0.

---

## 58. Contrat de compatibilité Web

Un plugin ne s'auto-déclare pas compatible. Son descripteur commence à :

```text
web_status = untested
```

Le statut devient `verified` pour une matrice de versions donnée seulement après tests réussis. Le noyau doit respecter au minimum :

- dépendances pure Python ou fournies par un wheel compatible avec la version de Pyodide utilisée ;
- pas d'accès arbitraire au système de fichiers ;
- pas de thread Python requis ;
- aucune API Qt hors adaptateur Desktop ;
- résultats sous forme d'objets DataLab/Sigima standard ;
- paramètres représentables via `DataSet`.

Une dépendance compilée n'est donc pas interdite par principe, mais elle est utilisable uniquement si un wheel Pyodide compatible existe et si le budget de téléchargement/mémoire reste acceptable. Les entry points Desktop ne participent pas à cette vérification.

---

## 59. Ordre Desktop/Web

Pour chaque plugin :

```text
1. scientific core headless
2. Desktop integration
3. stabilize workflow
4. Web integration
```

Mais la compatibilité Web doit être vérifiée dès l'étape 1.

Il ne faut pas écrire un plugin Desktop dépendant de Qt puis tenter de le porter plusieurs mois plus tard.

La vérification initiale est un test de contrat et d'import du noyau sous Pyodide, pas une promesse de plugin complet. Le statut `verified` exige ensuite un E2E visible exécutant la recette et vérifiant au moins une sortie DOM, une trace ou des pixels, ainsi qu'un budget mémoire sur les données de démonstration.

---

## 60. Expérience Web cible

**Cible après stabilisation du plugin.** Depuis le site :

```text
Characterize a scientific camera
[Try online]
```

ouvre :

```text
DataLab-Web
+ plugin
+ sample workspace
+ relevant application selected
```

Même chose pour :

```text
Analyze 500 oscilloscope pulses
```

et :

```text
Inspect a digital radiograph
```

L'utilisateur arrive directement dans le cas d'usage.

Le deep link doit vérifier que le plugin et la version de recette demandés sont effectivement intégrés au bundle. À défaut, l'interface affiche une erreur explicite ; elle ne tente pas d'installer silencieusement un package Python externe.

---

## 61. Dépôts recommandés

Créer progressivement des dépôts publics indépendants :

```text
DataLab-Platform/
        datalab-plugin-template
    datalab-camera-characterization
    datalab-pulse-characterization
    datalab-radiographic-ndt
```

Seuls le template minimal et Camera sont créés au démarrage. Pulse suit le durcissement du template. Le dépôt NDT n'est créé comme produit qu'après franchissement du gate données/expert ; un spike synthétique peut rester temporaire jusque-là.

Éviter de les mettre dans le dépôt DataLab principal.

Objectifs :

- cycle de release indépendant ;
- contributions externes plus simples ;
- visibilité GitHub propre ;
- documentation spécialisée ;
- démonstration réelle de l'architecture plugin ;
- possibilité qu'un plugin développe sa propre communauté.

---

## 62. Structure commune des dépôts

```text
repository/
│
├── pyproject.toml
├── README.md
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md
│
├── src/
│   └── package/
│       ├── __init__.py
│       ├── params.py
│       ├── core/
│       │   ├── algorithms.py
│       │   ├── simulation.py
│       │   └── validation.py
│       ├── workflow/
│       │   ├── recipes.py
│       │   └── outcomes.py
│       ├── adapters/
│       │   ├── desktop.py
│       │   └── web.py
│       ├── io/
│       └── examples/
│
├── tests/
│   ├── unit/
│   ├── validation/
│   ├── workflow/
│   └── integration/
│
└── doc/
```

`core/` et `workflow/` n'importent ni Qt ni le shim Web. Les adaptateurs sont minces et peuvent diverger sur UI, async, I/O et packaging. Cette structure devient celle du générateur après validation par Camera, pas avant.

---

## 63. Stratégie de tests

### Niveau 1 — algorithmes

Tests purs NumPy/SciPy.

Aucune dépendance DataLab GUI.

### Niveau 2 — processing

Tester :

```text
Input SignalObj/ImageObj
→ computation function
→ expected result object
```

### Niveau 3 — workflow

Tester une campagne entière sans interface :

```text
synthetic dataset
→ recipe
→ consolidated results
```

Vérifier aussi l'échec sans mutation du workspace, l'annulation, l'objet ancre, le `RecipeRunRecord`, les sorties multi-panneaux et le round-trip HDF5.

### Niveau 4 — intégration Desktop

Tester :

- chargement du plugin ;
- actions ;
- paramètres ;
- production des objets ;
- migration d'activation par ID ;
- découverte par entry point dans une installation Python ;
- reload/unload sans action ni traitement résiduel.

Les commandes DataLab et Sigima passent par leur environnement local :

```powershell
python scripts/run_with_env.py python -m pytest --ff
```

### Niveau 5 — Web

Tester les workflows représentatifs avec la même logique scientifique sous Pyodide, puis l'adaptateur dans le navigateur.

Pour toute modification DataLab-Web :

```powershell
npm test
```

Ajouter pytest `tests/python` pour les changements `src/runtime/*.py`, Playwright pour toute modification UI ou recette visible, et `npm run i18n:check` pour les chaînes utilisateur. Un test qui appelle seulement `window.runtime` sans assertion visible ne valide pas l'intégration UI.

---

## 64. Validation scientifique

Les tests logiciels ne suffisent pas.

Chaque plugin doit avoir un dossier :

```text
tests/validation/
```

avec :

- justification des équations ;
- références scientifiques ;
- jeux synthétiques ;
- résultats de référence ;
- tolérances ;
- comparaison éventuelle à d'autres implémentations.

Pour le CND, séparer clairement :

```text
algorithmic validation
```

de :

```text
inspection procedure validation
```

Le second nécessite une expertise métier spécifique.

---

## 65. Documentation de chaque plugin

Chaque plugin doit disposer de quatre niveaux.

### Landing page

Répond en moins de 30 secondes à :

> Pourquoi utiliser ce plugin ?

### Quickstart

Objectif :

```text
first useful result in a few minutes
```

### Tutorial métier

Explique :

- données ;
- protocole ;
- paramètres ;
- interprétation.

### Reference

API Python complète.

---

## 66. Démonstrations

Chaque plugin doit posséder une démo conçue comme un produit, pas comme un test technique.

### Démonstration Camera

> Compare two camera gain settings and identify which provides the best usable dynamic range.

### Démonstration Pulse

> Analyze 500 acquisitions and identify unstable and saturated shots.

### Démonstration NDT

> Detect, review and measure porosity indications in a weld radiograph.

Ces phrases doivent devenir des points d'entrée du site DataLab.

---

## 67. Feuille de route de développement

### Phase 0 — Recaler la stratégie et les gates

```text
0.1  publier ce document corrigé et un ADR du SDK minimal
0.2  capturer les baselines plugin, HDF5, Camera et Pulse
0.3  rechercher en parallèle données Camera/Pulse et expert + corpus CND
0.4  définir les conditions alpha/stable sans inventer de budgets
```

L'absence d'actifs externes ne bloque pas les prototypes synthétiques. Elle bloque les revendications scientifiques et releases stables correspondantes.

---

### Phase 1 — SDK minimal nécessaire à Camera

```text
1.1  stable plugin ID, collisions et migration des noms activés
1.2  feature_id, owner_plugin_id et suppression symétrique
1.3  lifecycle load/execute/reload/execute/unload testé
1.4  découverte Desktop/pip par entry point en complément de datalab_*
1.5  RecipeDescriptor, RecipeOutcome et diagnostics structurés
1.6  RecipeRunner transactionnel et contrat d'objet ancre
1.7  RecipeRunRecord local aux sorties
1.8  PluginExample via importlib.resources
1.9  IDs/origins alignés et tests de contrat DataLab-Web
```

Cette phase ne construit ni moteur graphique de workflow, ni métadonnées de groupe, ni historique global.

---

### Phase 2 — Squelette minimal et Camera headless

```text
2.1  commande datalab-plugin create et template volontairement mince
2.2  dépôt Camera séparant core, workflow et adaptateurs
2.3  simulateur déterministe avec vérité terrain
2.4  caractérisation relative en DN et diagnostics d'entrée
2.5  moyenne/variance incrémentales ou par blocs
2.6  RecipeOutcome avec courbe ancre, images utiles et table
2.7  validation synthétique, cas invalides et benchmarks mémoire/temps
```

Camera sert à découvrir les lacunes du SDK et du template ; elle ne doit pas attendre qu'ils soient supposés complets.

---

### Phase 3 — Camera Desktop et gate Alpha

```text
3.1  adaptateur Desktop mince et formulaire DataSet
3.2  sélection/assignation de rôles et commit multi-panneaux
3.3  quickstart packagé sans Python
3.4  installation wheel + entry point, reload et round-trip HDF5
3.5  gate Alpha : synthétique fiable, budget mémoire tenu, UX complète
```

Sans données réelles documentées et relecture scientifique, le résultat reste un alpha explicitement non métrologique.

---

### Phase 4 — Durcissement, spatial et Camera Web

```text
4.1  durcir le générateur à partir du retour Camera
4.2  étendre les cartes/profils/distributions et pixels candidats
4.3  adaptateur Web, bundling explicite et I/O navigateur
4.4  E2E visible courbe + carte + tableau et budget mémoire Pyodide
4.5  web_status verified uniquement après cette validation
4.6  gate Stable : données réelles documentées + revue scientifique
```

Les fonctions métrologiques avancées et la comparaison de caméras restent des jalons indépendants.

---

### Phase 5 — Pulse comme test de généralité signal

```text
5.1  générer Pulse depuis le template durci
5.2  réutiliser PulseFeatures et ajouter batch, intégrale/SNR et diagnostics
5.3  simulateur 500 tirs et flags explicables
5.4  alignement simple puis benchmark CPython/Pyodide
5.5  Desktop puis Web avec les mêmes gates que Camera
5.6  différer multi-voies et comparaison de configurations
```

---

### Phase 6 — Expérience Applications

Ne démarrer ce chantier qu'après existence de Camera et Pulse réellement utilisables.

```text
6.1  vue Applications consommant capacités, recettes et exemples
6.2  Start analysis, Open example et documentation
6.3  deep links DataLab-Web avec validation plugin/version
```

---

### Phase 7 — NDT radiographique sous gate externe

```text
7.1  obtenir expert CND et corpus exploitable annoté
7.2  spike synthétique limité tant que le gate n'est pas franchi
7.3  modèle de candidat et revue humaine persistante
7.4  pipeline compact-indication validé synthétique puis réel
7.5  chantier DICOM/DICONDE séparé
7.6  Desktop puis Web après validation du protocole
```

Sans expert et données, aucune roadmap produit NDT n'est engagée au-delà du spike.

---

### Phase 8 — Écosystème après preuve d'usage

Mesurer d'abord ouverture des exemples, réutilisation sur données utilisateur, demandes métier et contributions. N'ajouter index public, badges et processus de soumission qu'après preuve d'une réutilisation au-delà de la démonstration. Aucune marketplace complexe ni installation arbitraire depuis l'UI n'est prévue à ce stade.

---

## 68. Ordre de réalisation des plugins

Je conserverais cet ordre.

### 1. Camera & Detector Characterization

C'est le meilleur test architectural du concept.

Il requiert :

- traitement d'images ;
- séries ;
- métadonnées ;
- résultats tabulaires attachés à un objet ancre ;
- courbes ;
- cartes ;
- protocole métier ;
- validation scientifique.

Il permet donc au SDK minimal de rencontrer tôt ses principaux problèmes, sans prétendre les résoudre tous avant le pilote.

---

### 2. Pulse & Transient Characterization

Il permet ensuite de vérifier que l'infrastructure n'est pas trop orientée image.

Il exercera :

- `SignalObj` ;
- nombreuses acquisitions ;
- alignement ;
- batch ;
- qualité ;
- résultats agrégés.

Une grande partie du moteur scientifique est déjà disponible dans Sigima.

---

### 3. Radiographic NDT Analysis

Sous réserve du gate externe, il teste enfin le modèle sur un domaine industriel beaucoup plus spécialisé.

Il apportera :

- import métier ;
- métadonnées ;
- workflow avec validation humaine ;
- ROI et annotations ;
- détection ;
- inspection quantitative ;
- confidentialité / usage offline.

C'est un test exigeant de la capacité de DataLab à devenir le socle d'une application métier, mais il ne doit pas servir de pilote architectural tant que données et expertise manquent.

---

## 69. Ce qui ne doit pas être réalisé dans la première phase

Éviter :

- moteur graphique générique de workflows ;
- marketplace complète ;
- installation arbitraire de plugins depuis l'interface ;
- nouveau modèle d'objet Campaign ;
- groupes imbriqués ou métadonnées de groupe ;
- importeur générique de campagnes avant Camera et Pulse ;
- fork spécialisé de DataLab ;
- dépendance Qt dans les algorithmes ;
- support CND de toutes les modalités ;
- certification automatique EMVA ;
- décision automatique de conformité CND ;
- IA générative dans les trois plugins ;
- apprentissage automatique pour la première détection de défauts ;
- historique global ou manifeste de workflow spécifique aux plugins ;
- moteur générique de rapport PDF.

Ces éléments augmenteraient considérablement le périmètre sans tester l'hypothèse principale :

> **Un workflow métier prêt à l'emploi peut-il réellement attirer de nouveaux utilisateurs vers DataLab ?**

---

## 70. Critères de succès techniques du SDK

Le SDK minimal sera prêt pour l'expérimentation Camera lorsqu'il sera possible de :

1. identifier un plugin indépendamment de son nom et migrer l'ancienne activation ;
2. l'installer avec `pip` et le découvrir par entry point sur Desktop ;
3. conserver la découverte `datalab_*` pour développement/standalone ;
4. enregistrer des traitements avec un propriétaire et refuser les collisions ;
5. les retirer sans fuite lors du hot-reload et de l'unload ;
6. exécuter transactionnellement une recette headless ;
7. attacher résultats et `RecipeRunRecord` aux bonnes sorties ;
8. ouvrir un exemple provenant d'une ressource de package ;
9. utiliser uniquement des objets DataLab standards ;
10. partager noyau et workflow sans Qt entre adaptateurs ;
11. tester automatiquement le workflow et son round-trip HDF5.

La commande de génération mince peut évoluer en parallèle après stabilisation des descripteurs. La maturité générale du SDK n'est pas un prérequis déclaré pour démarrer Camera.

---

## 71. Critères de succès produit

Pour chaque plugin :

### Découverte

Un utilisateur comprend sa fonction depuis le nom et une capture d'écran.

### Premier usage

Il peut obtenir un résultat utile à partir du dataset exemple sans lire toute la documentation.

### Réutilisation

Il peut appliquer le même workflow à ses propres données.

### Traçabilité

Chaque objet produit porte une trace locale de recette suffisante pour identifier entrées, paramètres, versions et sorties liées.

### Automatisation

Le même workflow peut être invoqué en Python.

### Portabilité

Le noyau scientifique et le workflow sont utilisables sur Desktop ; la portabilité Web n'est acquise qu'après exécution visible et mesurée sous Pyodide pour la matrice de versions annoncée.

---

## 72. Critères de succès communautaire

Le véritable test ne sera pas le nombre de lignes de code.

Il faudra observer :

- téléchargements des plugins ;
- ouverture des démonstrations Web ;
- utilisateurs revenant sur le même plugin ;
- issues métier ouvertes par des utilisateurs ;
- formats instrumentaux demandés ;
- demandes de nouvelles recettes ;
- contributions externes ;
- jeux de données proposés ;
- citations ou usages dans des publications ;
- apparition de plugins tiers utilisant le même SDK.

Un signal particulièrement important serait :

> un utilisateur externe développe une recette, un importeur ou une extension pour l'un des trois plugins sans intervention directe de l'équipe DataLab.

---

## 73. Critères d'arrêt

Il faut aussi définir ce qui constituerait un résultat négatif.

Si un plugin :

- possède une excellente démonstration ;
- est facilement accessible ;
- est présenté à son public cible ;
- reçoit des essais réels ;
- mais n'est jamais réutilisé sur des données utilisateur ;

alors il ne faut pas continuer à ajouter des fonctionnalités dans l'espoir de créer artificiellement une demande.

Le workflow devra être réévalué ou abandonné.

---

## 74. Relation avec la stratégie communautaire

Ces plugins et le générateur doivent fonctionner ensemble.

Le générateur cible réduit la friction pour produire des extensions. Il commence comme squelette minimal, puis ses garanties de packaging, tests et documentation sont durcies à partir de Camera et Pulse. L'objectif de création en quelques minutes doit être mesuré sur le générateur livré, pas affirmé à partir de l'intention de roadmap.

Les trois plugins métier donnent ensuite au système une raison d'exister :

```text
Plugin SDK
     ↓
official métier plugins
     ↓
examples
     ↓
users
     ↓
new requirements
     ↓
external plugins
```

Sans applications concrètes, améliorer le SDK ne suffira pas.

Inversement, sans SDK accessible, les applications resteront exclusivement développées par l'équipe DataLab.

---

## 75. Vision cible

À terme, la plateforme devrait pouvoir être présentée sur son site non seulement comme :

> DataLab — signal and image processing platform

mais directement comme :

```text
What do you want to do?

[ Characterize a camera ]
[ Analyze repeated pulses ]
[ Inspect a radiograph ]

or

[ Open the general DataLab workspace ]
```

Chaque bouton ouvre la même plateforme.

Ce qui change est le point d'entrée.

---

## 76. Résultat architectural recherché

À l'issue des trois plugins, l'ajout d'un quatrième domaine comme :

```text
Spectroscopy
Laser Diagnostics
Optical Imaging Quality
Measurement Calibration
Particle Analysis
```

ne doit plus nécessiter de réfléchir à :

- comment déclarer le plugin ;
- comment le packager ;
- comment enregistrer ses traitements ;
- comment créer une recette ;
- comment distribuer un exemple ;
- comment construire ses tests ;
- comment le faire apparaître comme application.

Ces problèmes doivent être largement standardisés, documentés et couverts par des tests de contrat. Le SDK reste évolutif : un quatrième plugin peut révéler un besoin légitime sans invalider l'approche.

Le quatrième plugin doit donc essentiellement poser deux questions :

> Quel problème métier résolvons-nous ?

et :

> Quels calculs scientifiques devons-nous implémenter ?

C'est ce passage d'une logique de « framework interne » à une véritable **plateforme d'applications scientifiques extensible** qui constitue le résultat principal de cette feuille de route.

---

## 77. Matrice de revue factuelle

Cette matrice conserve la trace des principales corrections apportées au plan. « Preuve code » désigne le symbole propriétaire observé dans la base au moment de la révision.

| Section | Affirmation initiale | Correction | Preuve code |
| --- | --- | --- | --- |
| 2, 14 | Une branche de provenance globale devait porter l'historique des plugins. | Conserver la provenance unitaire existante et ajouter un `RecipeRunRecord` local aux sorties, sans manifeste global. | `datalab/gui/processor/base.py::ProcessingParameters` |
| 4, 6 | Le cycle de vie devait notamment apprendre à nettoyer les actions. | Le reload nettoie déjà actions et sous-menus ; le défaut porte sur l'ownership et le retrait des traitements. | `datalab/gui/main.py::reload_plugins`, `datalab/gui/actionhandler.py::clear_plugin_actions` |
| 5 | Le descripteur semblait déjà extensible avec de nombreuses métadonnées. | `PluginInfo` n'a que quatre champs et le registre/config utilisent le nom ; ID et migration sont des cibles. | `datalab/plugins.py::PluginInfo`, `PluginRegistry.get_plugin`, `datalab/config.py::plugins_enabled_list` |
| 6 | L'ordre proposé était register computations, actions, hooks. | Desktop appelle `register_hooks()` dans `register()`, puis la fenêtre crée les actions plus tard. | `datalab/plugins.py::PluginBase.register`, `datalab/gui/main.py::__create_plugins_actions` |
| 6 | Les traitements externes paraissaient symétriques. | Le registre est indexé par fonction et ne possède aucune suppression par ID ou propriétaire. | `datalab/gui/processor/base.py::ComputingFeature`, `BaseProcessor.add_feature` |
| 8, 9 | Un callable de recette et un chemin relatif suffisaient. | Slots typés, outcome, diagnostics, transaction, ancre et ressource de package sont nécessaires. | `datalab/adapters_metadata/base_adapter.py::BaseResultAdapter.add_to` |
| 10 | Les entry points pouvaient ordonner toutes les sources de plugins. | La convention `datalab_*` existe ; les entry points sont une cible Desktop/pip et les collisions d'ID sont fatales. | `datalab/plugins.py::discover_plugins` |
| 11 | Un générateur Contributor Experience était présenté comme existant. | Seuls une intention et des templates de tests existent ; `datalab-plugin create` est un livrable cible. | `datalab/tests/features/plugins/templates/`, `pyproject.toml` |
| 12, 13 | L'exemple utilisait des groupes imbriqués et des métadonnées de campagne au niveau groupe. | Les groupes sont plats, séparés par panneau et sans métadonnées ; les objets portent les clés namespacées. | `datalab/objectmodel.py::ObjectGroup`, `ObjectModel` |
| 14, 23, 46, 54 | Tables et géométries pouvaient être traitées comme sorties top-level. | Elles sont stockées dans les métadonnées d'un `SignalObj` ou `ImageObj` ancre. | `datalab/adapters_metadata/base_adapter.py::BaseResultAdapter.add_to` |
| 18 à 26 | Le workflow et les métriques Camera étaient décrits sans état d'implémentation. | Les primitives existent partiellement ; importeur, simulateur, métriques campagne et workflow restent à construire. | `../Sigima/sigima/proc/image/arithmetic.py`, `exposure.py`, `extraction.py`, `measurement.py` |
| 21 | L'écart-type semblait réutilisable sans contrainte de pile. | L'implémentation matérialise toutes les images et doit être remplacée/encadrée pour les grandes piles. | `../Sigima/sigima/proc/image/arithmetic.py::standard_deviation` |
| 28 à 36 | Intégrale, SNR et campagne Pulse semblaient appartenir au socle existant. | `PulseFeatures` couvre amplitude, offset, polarité, temps et FWHM ; batch, intégrale/SNR consolidés et flags manquent. | `../Sigima/sigima/tools/signal/pulse.py::PulseFeatures`, `extract_pulse_features` |
| 40 | Le support DICOM pouvait servir de base DICONDE en préservant les informations. | Le lecteur actuel retourne principalement les pixels ; mapping des tags et DICONDE forment un chantier séparé. | `../Sigima/sigima/io/image/funcs.py::imread_dicom`, `formats.py::DICOMImageFormat` |
| 44 à 46 | Résultat géométrique, annotation et ROI étaient assimilés. | Ce sont des représentations distinctes à relier par un ID de candidat stable. | `../Sigima/sigima/objects/scalar/`, `datalab/adapters_plotpy/annotations.py` |
| 53 à 56 | Importeur, statuts, consolidation et rapports pouvaient être lus comme APIs communes. | Ce sont des cibles à extraire seulement après deux cas réels ; seule l'ancre de résultat est requise en P0. | Absence de `CampaignImporter`; adaptateurs de résultats existants |
| 57 à 60 | Desktop et Web étaient décrits comme un même système de plugins. | Web fournit un shim partiel, async et chargé explicitement, avec registres par origine déjà nettoyables. | `../DataLab-Web/src/runtime/dlplugins/datalab/registries.py`, `plugins.py`, `gui/processor/base.py` |
| 58 | La compatibilité Web était un booléen et impliquait du pur Python. | Elle est vérifiée par tests ; une dépendance compilée exige un wheel Pyodide compatible. | `../DataLab-Web/src/runtime/runtime.ts::PYODIDE_VERSION`, `shims/registry.ts::PACKAGE_VERSION_SOURCES` |
| 61 à 67 | SDK et générateur complets précédaient les pilotes. | Un SDK minimal et un squelette précèdent Camera ; le retour Camera durcit ensuite le générateur et le port Web. | Dépendances et lacunes listées dans les lignes précédentes |
| 67, NDT | Le développement NDT suivait automatiquement Camera et Pulse. | Données réelles exploitables et expert CND sont des gates avant tout produit. | Aucun modèle de revue CND ni corpus métier dans DataLab/Sigima |
