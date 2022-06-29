import superimport

import daft

# Colors.
p_color = {"ec": "#46a546"}
s_color = {"ec": "#f89406"}
r_color = {"ec": "#dc143c"}

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

pgm = daft.PGM((10, 8))

pgm.add_node("sigma_y", r"$\sigma_y ^2$", 1, 6, plot_params=s_color)

j = 7
# y and mu_y
for i in range(3):
    pgm.add_node("y_{}".format(i + 1), r"$y_{}$".format(i + 1), 2, j - i, observed=True)
    pgm.add_node("mu_y_{}".format(i + 1), r"$\mu_{} ^y$".format(i + 1), 3, j - i, plot_params=p_color)

# X_merged
for i in range(3):
    for j in range(2):
        pgm.add_node("X_merged[{}][{}]".format(i + 1, j + 1), r"$X{}{} ^M$".format(i + 1, j + 1).translate(SUB), 5 + j,
                     7 - i, plot_params=s_color)
# X_imputed
for i in range(3):
    for j in range(2):
        if i != j:
            pgm.add_node("X_impute[{}][{}]".format(i + 1, j + 1), r"$X{}{} ^M$".format(i + 1, j + 1).translate(SUB),
                         8 + j, 6 - i, observed=True)
        else:
            pgm.add_node("X_impute[{}][{}]".format(i + 1, j + 1), r"$X{}{} ^M$".format(i + 1, j + 1).translate(SUB),
                         8 + j, 6 - i, plot_params=r_color)

# Per sample
for i in range(3):
    pgm.add_node("sample_{}".format(i + 1), "", 5.5, 7 - i, scale=1.4, aspect=3, shape='ellipse')

# per feature
for i in range(2):
    pgm.add_node("feature_{}".format(i + 1), "", 5 + i, 6, scale=5.2, aspect=1.5 / 5.2, shape='ellipse')

# Weights
pgm.add_node("a", r"$a$", 4, 4, plot_params=s_color)
pgm.add_node("beta", r"$\beta$", 2, 4, plot_params=s_color)

# mu_x
pgm.add_node("mu_1", r"$\mu_1 ^X$", 5, 3, plot_params=p_color)
pgm.add_node("mu_2", r"$\mu_2 ^X$", 6, 3, plot_params=p_color)

# covariance_x
pgm.add_node("covariance", r"$Σ ^X$", 5.5, 2, plot_params=s_color)

# correlation_x, sigma_x, standard_normal
pgm.add_node("Rho", r"$R ^x$", 5, 1, plot_params=p_color)
pgm.add_node("sigma_x", r"$\sigma ^X$", 6, 1, plot_params=p_color)
pgm.add_node("N(0,1)", "N(0,1)", 8.5, 5, fixed=True, aspect=1.0, offset=[0.2, -2.3])

# edges
for i in range(3):
    pgm.add_edge("sigma_y", "y_{}".format(i + 1))  # sigma_y -> y
    pgm.add_edge("a", "mu_y_{}".format(i + 1))  # a -> mu_y
    pgm.add_edge("beta", "mu_y_{}".format(i + 1))  # beta -> mu_y
    pgm.add_edge("sample_{}".format(i + 1), "mu_y_{}".format(i + 1))  # x -> mu_y
    pgm.add_edge("mu_y_{}".format(i + 1), "y_{}".format(i + 1))  # mu_y -> y

pgm.add_plate([4.4, 4, 2.2, 3.4], label="X_merged", shift=-0.1)
pgm.add_plate([7.4, 3, 2.2, 3.4], label="X_impute", shift=-0.1)

pgm.add_edge("Rho", "covariance")  # correlation -> covariance
pgm.add_edge("sigma_x", "covariance")  # sigma_x -> covariance

# mu_x -> feature_x
pgm.add_edge("mu_1", "feature_1")
pgm.add_edge("mu_2", "feature_2")

# X_merged -> X_impute (only two edges are added for visible purpose)
pgm.add_edge("X_merged[1][1]", "X_impute[1][1]")
pgm.add_edge("X_merged[1][2]", "X_impute[1][2]")

# N(0,1)(prior) -> imputed values of X
pgm.add_edge("N(0,1)", "X_impute[1][1]")
pgm.add_edge("N(0,1)", "X_impute[2][2]")
pgm.add_edge("covariance", "sample_3")

# labels
pgm.add_node("missing", "", 2, 3, 1, plot_params=r_color)
pgm.add_text(2.25, 3 - 0.08, " = missing data")
pgm.add_node("observed", "", 2, 2, 1, observed=True)
pgm.add_text(2.25, 2 - 0.08, " = observed variable")
pgm.add_text(1, 0.5, "All other than observed variables are latent variables")

pgm.render()
pgm.savefig('../figures/linreg_impute_pgm.pdf')
pgm.show()
