import time
import numpy as np
import matplotlib.pyplot as plt

def compute_and_plot_06(Av_mul, pre_mul, cg, pcg, m, tol, maxit):
    """
    Diese Routine ruft beide Loesungsverfahren (CG und PCG) auf.
    Die maximale Iterationszahl 'maxit', die absolute Residuuen-Genauigkeit
    'tol' und die Komplexitaet der Aufgabe 'k' (Variablenzahl = 2^k+1)
    ist fuer alle Verfahren gleichermassen anzuwenden.
    Die (Zeilensummen-)Norm der Systemmatrix ist aufgabenbedingt stets 4.0.
    """
    #
    #  time und numpy.linalg initialisieren dauert SEHR lang!
    #  Die folgenden Zeilen sind eingefuegt, damit die Laufzeit-Tabelle in der
    #  ersten Zeile (m==3) die Initialisierungszeiten der Bibliotheken nicht
    #  mitzaehlt; bitte NICHT entfernen!
    #
    _ = time.perf_counter()
    _ = np.ones(1)

    #  Aufgabe initialisieren:
    init_tic = time.perf_counter()
    pn = 2**m
    b = -np.ones(pn + 1)
    half_point = int(pn * 0.5 + 0.5)
    b[half_point] = 1.0

    A = 2.0 * np.eye(pn + 1) - np.diag(np.ones(pn), k=-1) - np.diag(np.ones(pn), k=1)
    ref_solution = np.linalg.solve(A, b)
    init_time = time.perf_counter() - init_tic

    #CG
    cg_tic = time.perf_counter()
    cg_solution, cg_residuals = cg(b, Av_mul, tol, maxit)
    cg_time = time.perf_counter() - cg_tic
    if (type(cg_solution) == np.ndarray):
        cg_norm = np.max(np.abs(ref_solution - cg_solution))

    #PCG
    pcg_tic = time.perf_counter()
    pcg_solution, pcg_residuals = pcg(b, Av_mul, pre_mul, tol, maxit)
    pcg_time = time.perf_counter() - pcg_tic
    pcg_norm = np.max(np.abs(ref_solution - pcg_solution))

    #Plots
    cg_xx  = np.arange(1, cg_residuals.shape[0] + 1)
    pcg_xx = np.arange(1, pcg_residuals.shape[0] + 1)
    xl = np.linspace(-1.0, 1.0, cg_solution.shape[0])
    plt.figure()

    plt.subplot(221)
    plt.title("CG - Residuen")
    plt.xlim(1.0, cg_xx[-1])
    plt.plot(cg_xx, np.log10(cg_residuals), "b")

    plt.subplot(223)
    plt.title("CG - Fehler")
    plt.plot(xl, ref_solution - cg_solution, "r")

    plt.subplot(222)
    plt.title("PCG - Residuen")
    plt.xlim(1.0, pcg_xx[-1])
    plt.plot(pcg_xx, np.log10(pcg_residuals), "b")

    plt.subplot(224)
    plt.title("PCG - Fehler")
    plt.plot(xl, ref_solution - pcg_solution, "r")

    plt.suptitle(f"CG- vs. PCG-Verfahren  @  m = {m},   n = {2**m + 1}")
    plt.tight_layout()
    
    return (
        cg_solution,
        cg_residuals,
        cg_norm,
        cg_time,
        pcg_solution,
        pcg_residuals,
        pcg_norm,
        pcg_time,
        init_time,
    )
