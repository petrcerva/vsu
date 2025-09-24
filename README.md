# Podklady pro předmět Vybrané statě ze strojového učení

Úlohy jsou připravené ve formě jupyter notebooků jazyka Python 3.x především s využitím knihoven numpy a matplotlib.

## Instalace

Nejjednodušší cesta, jak vše zprovoznit na vlastním počítači s Windows 10 či Linuxem, je:

1. Instalace: [64-bitová 3.9 verze distribuce Miniconda](https://docs.conda.io/en/latest/miniconda.html)
	- Při instalaci nevolte přidání spustitelných skriptů Anacondy do standardních cest. Po instalaci spusťte Anaconda Prompt a zadejte příkaz conda init, aby bylo možné ji používat i z běžného příkazového řádku (cmd.exe). V opačném případě bude pro práci vždy nutné spouštět Anaconda Prompt, která všechny potřebné cesty již obsahuje.
	- Neinstalujte do adresáře obsahujícího v cestě slova s diakritikou.
  - Před instalací odstraňte předchozí verze Pythonu.
2. Vytvoření prostředí pro předmět VSU v běžné příkazové řádce: conda create -n vsu python=3.10
3. Aktivace vytvořeného prostředí: conda activate vsu
4. Instalace modulů ze seznamu závislostí do vytvořeného prostředí:
   - jako conda balíky příkazem `pip install <balik>`,
   - pip install torch
   - pip install notebook   
   - pip install numpy   
   - pip install matplotlib
5. Spuštění jupyter notebooku v příkazové řádce: jupyter notebook (v případě problémů: python -n notebook)

## Podmínky zápočtu

- Pro získání zápočtu je nutné samostatně vypracovat a odevzdat v uvedeném termínu všechny úlohy nebo jejich části, které nejsou označeny jako bonusové.
- Za vypracování bonusových úloh nebo bonusových částí povinných úloh je možné získat plusové body ke zkoušce. Ty mohou významně zlepšit výslednou známku.
- **Neodevzdání úlohy v termínu stejně jako zcela či z podstatné části zkopírovaná úloha má za následek ztrátu nároku na zápočet.

## Jednotlivá cvičení

### 1. Úvod 
- Seznamení se s prostředím jupyter notebook:
- Rychlý úvod [youtube](https://www.youtube.com/watch?v=HW29067qVWk)
- Seznámení se s knihovnou NumPy:
- Rychlý úvod [numpy.org](https://numpy.org/doc/stable/user/quickstart.html) až do začátku části "Splitting one array into several smaller ones". 
- Rozdíly mezi Numpy a Matlabem [numpy.org](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html)
- Notebook: [VSU_01_INTRO_CZ.ipynb](VSU_01_INTRO_CZ.ipynb)
- **deadline: 25.9.2025 7:59**
 
### 2. Regrese analyticky
- Notebook: [USU_02_LR_LSE_CZ.ipynb](USU_02_LR_LSE_CZ.ipynb)
- Bonusová část: podúloha na exponenciální regresi
- **deadline: 2.10.2025 7:59**