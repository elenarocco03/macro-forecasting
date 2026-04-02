import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA


# ============================================================
# BIC PLOTS — Ridge and Lasso
# ============================================================

def plot_bic(alphas_ridge, bic_ridge_ipi, bic_ridge_cpi,
             alphas_lasso, bic_lasso_ipi, bic_lasso_cpi,
             lambda_ridge_ipi, lambda_ridge_cpi,
             lambda_lasso_ipi, lambda_lasso_cpi,
             save_path=None):
    """
    Plot BIC as a function of lambda for Ridge and Lasso (IPI and PCEPI).

    Parameters
    ----------
    alphas_ridge : array - grid of Ridge lambda values
    bic_ridge_ipi : list - BIC scores for Ridge IPI
    bic_ridge_cpi : list - BIC scores for Ridge PCEPI
    alphas_lasso : array - grid of Lasso lambda values
    bic_lasso_ipi : list - BIC scores for Lasso IPI
    bic_lasso_cpi : list - BIC scores for Lasso PCEPI
    lambda_ridge_ipi : float - optimal Ridge lambda for IPI
    lambda_ridge_cpi : float - optimal Ridge lambda for PCEPI
    lambda_lasso_ipi : float - optimal Lasso lambda for IPI
    lambda_lasso_cpi : float - optimal Lasso lambda for PCEPI
    save_path : str or None - path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(r"BIC as a function of $\lambda$", fontsize=14)

    # Ridge IPI
    axes[0, 0].plot(np.log10(alphas_ridge), bic_ridge_ipi)
    axes[0, 0].axvline(np.log10(lambda_ridge_ipi), color='red', linestyle='--',
                       label=rf'$\lambda$ = {lambda_ridge_ipi}')
    axes[0, 0].set_title(r"Ridge — $\Delta \log$ IPI")
    axes[0, 0].set_xlabel(r"$\log_{10}(\lambda)$")
    axes[0, 0].set_ylabel("BIC")
    axes[0, 0].legend()

    # Ridge PCEPI
    axes[0, 1].plot(np.log10(alphas_ridge), bic_ridge_cpi)
    axes[0, 1].axvline(np.log10(lambda_ridge_cpi), color='red', linestyle='--',
                       label=rf'$\lambda$ = {lambda_ridge_cpi}')
    axes[0, 1].set_title(r"Ridge — $\Delta^2 \log$ PCEPI")
    axes[0, 1].set_xlabel(r"$\log_{10}(\lambda)$")
    axes[0, 1].set_ylabel("BIC")
    axes[0, 1].legend()

    # Lasso IPI
    axes[1, 0].plot(np.log10(alphas_lasso), bic_lasso_ipi)
    axes[1, 0].axvline(np.log10(lambda_lasso_ipi), color='red', linestyle='--',
                       label=rf'optimal $\lambda$ = {lambda_lasso_ipi:.4f}')
    axes[1, 0].set_title(r"Lasso — $\Delta \log$ IPI")
    axes[1, 0].set_xlabel(r"$\log_{10}(\lambda)$")
    axes[1, 0].set_ylabel("BIC")
    axes[1, 0].legend()

    # Lasso PCEPI
    axes[1, 1].plot(np.log10(alphas_lasso), bic_lasso_cpi)
    axes[1, 1].axvline(np.log10(lambda_lasso_cpi), color='red', linestyle='--',
                       label=rf'optimal $\lambda$ = {lambda_lasso_cpi:.4f}')
    axes[1, 1].set_title(r"Lasso — $\Delta^2 \log$ PCEPI")
    axes[1, 1].set_xlabel(r"$\log_{10}(\lambda)$")
    axes[1, 1].set_ylabel("BIC")
    axes[1, 1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ============================================================
# LASSO COEFFICIENT HEATMAP
# ============================================================

def lasso_coef_path(X, y, alphas):
    """
    Compute Lasso coefficients for each alpha (penalization parameter)using warm start.

    Parameters
    ----------
    X : array (n, p) - standardized predictor matrix
    y : array (n,)   - standardized target variable
    alphas : array   - grid of lambda values

    Returns
    -------
    coefs : array (p, n_alphas) - coefficients for each lambda
    """
    coefs = []
    model = Lasso(fit_intercept=False, max_iter=100000, warm_start=True)
    for alpha in alphas:
        model.set_params(alpha=alpha)
        model.fit(X, y)
        coefs.append(model.coef_.copy())
    return np.array(coefs).T

# ============================================================
# SCREE PLOT — PCA
# ============================================================

def plot_scree(ev, r=6, save_path=None):
    """
    Plot scree plot and cumulative explained variance for PCA.

    Parameters
    ----------
    ev : array - explained variance ratios from PCA
    r : int - number of components to highlight (elbow)
    save_path : str or None
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(range(1, len(ev) + 1), ev * 100, 'o-')
    axes[0].axvline(x=r, color='red', linestyle='--', label=f'r = {r}')
    axes[0].set_title("Scree Plot")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].legend()

    axes[1].plot(range(1, len(ev) + 1), np.cumsum(ev) * 100, 'o-')
    axes[1].axhline(y=90, color='red', linestyle='--', label='90%')
    axes[1].axvline(x=r, color='orange', linestyle='--', label=f'r = {r}')
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].set_xlabel("Principal Component")
    axes[1].set_ylabel("Cumulative (%)")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

# ============================================================
# PC OVER TIME
# ============================================================

def PCA_components(n_PC, X_train, X_full, dates_full, save_path=None):

    import pandas as pd
    
    pca = PCA(n_components=n_PC)
    pca.fit(X_train)           # fit solo su training
    F = pca.transform(X_full)  # trasforma tutto il dataset

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"First {n_PC} Principal Components", fontsize=14)

    NBER_RECESSIONS = [
    ("1960-04", "1961-02"),
    ("1969-12", "1970-11"),
    ("1973-11", "1975-03"),
    ("1980-01", "1980-07"),
    ("1981-07", "1982-11"),
    ("1990-07", "1991-03"),
    ("2001-03", "2001-11"),
    ("2007-12", "2009-06"),
    ("2020-02", "2020-04"),
]

    for i, ax in enumerate(axes.flatten()):
        ax.plot(dates_full[:len(F)], F[:, i])
        ax.set_title(f"PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)")
        ax.axhline(0, color='black', linewidth=0.5)
        for start, end in NBER_RECESSIONS:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.2, color='grey')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()