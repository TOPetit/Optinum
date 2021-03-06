@doc doc"""
Résolution des problèmes de minimisation sous contraintes d'égalités

# Syntaxe
```julia
Lagrangien_Augmente(algo,fonc,contrainte,gradfonc,hessfonc,grad_contrainte,
			hess_contrainte,x0,options)
```

# Entrées
  * **algo** 		   : (String) l'algorithme sans contraintes à utiliser:
    - **"newton"**  : pour l'algorithme de Newton
    - **"cauchy"**  : pour le pas de Cauchy
    - **"gct"**     : pour le gradient conjugué tronqué
  * **fonc** 		   : (Function) la fonction à minimiser
  * **contrainte**	   : (Function) la contrainte [x est dans le domaine des contraintes ssi ``c(x)=0``]
  * **gradfonc**       : (Function) le gradient de la fonction
  * **hessfonc** 	   : (Function) la hessienne de la fonction
  * **grad_contrainte** : (Function) le gradient de la contrainte
  * **hess_contrainte** : (Function) la hessienne de la contrainte
  * **x0** 			   : (Array{Float,1}) la première composante du point de départ du Lagrangien
  * **options**		   : (Array{Float,1})
    1. **epsilon** 	   : utilisé dans les critères d'arrêt
    2. **tol**         : la tolérance utilisée dans les critères d'arrêt
    3. **itermax** 	   : nombre maximal d'itération dans la boucle principale
    4. **lambda0**	   : la deuxième composante du point de départ du Lagrangien
    5. **mu0,tho** 	   : valeurs initiales des variables de l'algorithme

# Sorties
* **xmin**		   : (Array{Float,1}) une approximation de la solution du problème avec contraintes
* **fxmin** 	   : (Float) ``f(x_{min})``
* **flag**		   : (Integer) indicateur du déroulement de l'algorithme
   - **0**    : convergence
   - **1**    : nombre maximal d'itération atteint
   - **(-1)** : une erreur s'est produite
* **niters** 	   : (Integer) nombre d'itérations réalisées

# Exemple d'appel
```julia
using LinearAlgebra
f(x)=100*(x[2]-x[1]^2)^2+(1-x[1])^2
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
algo = "gct" # ou newton|gct
x0 = [1; 0]
options = []
contrainte(x) =  (x[1]^2) + (x[2]^2) -1.5
grad_contrainte(x) = [2*x[1] ;2*x[2]]
hess_contrainte(x) = [2 0;0 2]
output = Lagrangien_Augmente(algo,f,contrainte,gradf,hessf,grad_contrainte,hess_contrainte,x0,options)
```
"""

include("Algorithme_De_Newton.jl")
include("Regions_De_Confiance.jl")

function Lagrangien_Augmente(algo,fonc::Function,contrainte::Function,gradfonc::Function,
	hessfonc::Function,grad_contrainte::Function,hess_contrainte::Function,x0,options)

	if options == []
		epsilon = 1e-8
		tol = 1e-5
		itermax = 1000
		lambda0 = 2
		mu0 = 100
		tho = 2
	else
		epsilon = options[1]
		tol = options[2]
		itermax = options[3]
		lambda0 = options[4]
		mu0 = options[5]
		tho = options[6]
	end

    n = length(x0)
    xmin = zeros(n)
	fxmin = 0
	flag = -1
    iter = 0
    

    continuer = true
    xk = x0
    lambdak = mu0 * contrainte(x0)
    muk = mu0
    beta = 0.9
    alpha = 0.1
    eps0 = epsilon
    eta_bis0 = 0.1258925
    etak = eta_bis0 / (mu0^alpha)
  
    while continuer

        L_A(x) = fonc(x) + transpose(lambdak) * contrainte(x) + 0.5 * muk * norm(contrainte(x))^2
        grad_L_A(x) = gradfonc(x) + transpose(lambdak) * grad_contrainte(x) + muk * transpose(contrainte(x)) * grad_contrainte(x)
        hess_L_A(x) = hessfonc(x) + transpose(lambdak) * hess_contrainte(x) + muk * grad_contrainte(x) * transpose(grad_contrainte(x)) + hess_contrainte(x) * contrainte(x)

        if algo == "newton"
            xmin, _, _, _ = Algorithme_De_Newton(L_A, grad_L_A, hess_L_A, xk, [100, epsilon, epsilon])
        else
            xmin, _, _, _ = Regions_De_Confiance(algo, L_A, grad_L_A, hess_L_A, xk, [10, 0.75, 2, 0.25, 0.75, 2, 5000, epsilon, epsilon])
        end

        if norm(xk - xmin) < tol || norm(xk - xmin) / max(norm(xk), norm(xmin)) < tol
            # Convergence
            continuer = false
            flag = 0
        end

        if norm(contrainte(xmin)) <= etak
            lambdak = lambdak + muk * contrainte(xmin)
            epsilon = epsilon / muk
            etak = etak / (muk^beta)
        else
            muk = muk * tho
            espsilon = eps0 / muk
            etak = eta_bis0 / (muk^alpha)
        end

        xk = xmin
        iter = iter + 1

        if iter >= itermax
            continuer = false
            flag = 1
        end

    end

    xmin = xk
    fxmin = fonc(xmin)

	return xmin, fxmin, flag, iter, lambdak, muk
end
