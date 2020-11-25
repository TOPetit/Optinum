@doc doc"""
Approximation de la solution du problème ``\min_{x \in \mathbb{R}^{n}} f(x)`` en utilisant l'algorithme de Newton

# Syntaxe
```julia
xk,f_min,flag,nb_iters = Algorithme_de_Newton(f,gradf,hessf,x0,option)
```

# Entrées :
   * **f**       : (Function) la fonction à minimiser
   * **gradf**   : (Function) le gradient de la fonction f
   * **hessf**   : (Function) la Hessienne de la fonction f
   * **x0**      : (Array{Float,1}) première approximation de la solution cherchée
   * **options** : (Array{Float,1})
       * **max_iter**      : le nombre maximal d'iterations
       * **Tol_abs**       : la tolérence absolue
       * **Tol_rel**       : la tolérence relative

# Sorties:
   * **xmin**    : (Array{Float,1}) une approximation de la solution du problème  : ``\min_{x \in \mathbb{R}^{n}} f(x)``
   * **f_min**   : (Float) ``f(x_{min})``
   * **flag**     : (Integer) indique le critère sur lequel le programme à arrêter
      * **0**    : Convergence
      * **1**    : stagnation du xk
      * **2**    : stagnation du f
      * **3**    : nombre maximal d'itération dépassé
   * **nb_iters** : (Integer) le nombre d'itérations faites par le programme

# Exemple d'appel
```@example
using Optinum
f(x)=100*(x[2]-x[1]^2)^2+(1-x[1])^2
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
x0 = [1; 0]
options = []
xmin,f_min,flag,nb_iters = Algorithme_De_Newton(f,gradf,hessf,x0,options)
```
"""
function Algorithme_De_Newton(f::Function, gradf::Function, hessf::Function, x0, options)

    "# Si options == [] on prends les paramètres par défaut"
    if options == []
        max_iter = 100
        Tol_abs = sqrt(eps())
        Tol_rel = 1e-15
    else
        max_iter = options[1]
        Tol_abs = options[2]
        Tol_rel = options[3]
    end

    n = length(x0)
    xk = x0
    val = f(x0)
    nb_iters = 0
    continuer = true
    flag = -1
    while continuer
        
        if norm(gradf(xk)) < eps()
            continuer = false
            flag = 0
            break
        end
        
        xk_prec = xk
        val_prec = val
        dk = inv(hessf(xk)) * gradf(xk)
        #println(xk, " et ", dk)
        xk = xk - dk
        val = f(xk)
        norm_dk = norm(dk) #utile pour les tests suivant cas norm(xk - xk_prec) = norm(dk)

        nb_iters = nb_iters + 1

        if nb_iters >= max_iter
            continuer = false
            flag = 3
        end
        if (norm_dk < Tol_abs || norm_dk / min(norm(val), norm(val_prec)) < Tol_rel)
            continuer = false
            flag = 2
        end
        if (norm_dk < Tol_abs || norm_dk / min(norm(xk), norm(xk_prec)) < Tol_rel)
            continuer = false
            flag = 1
        end
    end
    xmin = xk
    f_min = val
    return xmin, f_min, flag, nb_iters
end

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        