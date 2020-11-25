@doc doc"""
Minimise une fonction en utilisant l'algorithme des régions de confiance avec
    - le pas de Cauchy
ou
    - le pas issu de l'algorithme du gradient conjugue tronqué

# Syntaxe
```julia
xk, nb_iters, f(xk), flag = Regions_De_Confiance(algo,f,gradf,hessf,x0,option)
```

# Entrées :

   * **algo**        : (String) string indicant la méthode à utiliser pour calculer le pas
        - **"gct"**   : pour l'algorithme du gradient conjugué tronqué
        - **"cauchy"**: pour le pas de Cauchy
   * **f**           : (Function) la fonction à minimiser
   * **gradf**       : (Function) le gradient de la fonction f
   * **hessf**       : (Function) la hessiene de la fonction à minimiser
   * **x0**          : (Array{Float,1}) point de départ
   * **options**     : (Array{Float,1})
     * **deltaMax**      : utile pour les m-à-j de la région de confiance
                      ``R_{k}=\left\{x_{k}+s ;\|s\| \leq \Delta_{k}\right\}``
     * **gamma1,gamma2** : ``0 < \gamma_{1} < 1 < \gamma_{2}`` pour les m-à-j de ``R_{k}``
     * **eta1,eta2**     : ``0 < \eta_{1} < \eta_{2} < 1`` pour les m-à-j de ``R_{k}``
     * **delta0**        : le rayon de départ de la région de confiance
     * **max_iter**      : le nombre maximale d'iterations
     * **Tol_abs**       : la tolérence absolue
     * **Tol_rel**       : la tolérence relative

# Sorties:

   * **xmin**    : (Array{Float,1}) une approximation de la solution du problème : ``min_{x \in \mathbb{R}^{n}} f(x)``
   * **fxmin**   : (Float) ``f(x_{min})``
   * **flag**    : (Integer) un entier indiquant le critère sur lequel le programme à arrêter
      - **0**    : Convergence
      - **1**    : stagnation du ``x``
      - **2**    : stagnation du ``f``
      - **3**    : nombre maximal d'itération dépassé
   * **nb_iters** : (Integer)le nombre d'iteration qu'à fait le programme

# Exemple d'appel
```julia
algo="gct"
f(x)=100*(x[2]-x[1]^2)^2+(1-x[1])^2
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
x0 = [1; 0]
options = []
xmin, fxmin, flag,nb_iters = Regions_De_Confiance(algo,f,gradf,hessf,x0,options)
```
"""

include("Pas_De_Cauchy.jl")
function Regions_De_Confiance(algo, f::Function, gradf::Function, hessf::Function, x0, options)

    if options == []
        deltaMax = 10
        gamma1 = 0.25
        gamma2 = 2.00
        eta1 = 0.25
        eta2 = 0.75
        delta0 = 2
        max_iter = 1000
        Tol_abs = sqrt(eps())
        Tol_rel = 1e-15
    else
        deltaMax = options[1]
        gamma1 = options[2]
        gamma2 = options[3]
        eta1 = options[4]
        eta2 = options[5]
        delta0 = options[6]
        max_iter = options[7]
        Tol_abs = options[8]
        Tol_rel = options[9]
    end

    n = length(x0)
    x_tard = zeros(n)

    flag = 0
    nb_iters = 0

    x_k = x0
    val = f(x0)
    delta_k = delta0

    if algo == "cauchy"

        # Condition d'arrêt de la boucle while
        continuer = true

        while continuer


            # Calcul du gradient et de la hessienne en x0
            g_k = gradf(x_k)
            H_k = hessf(x_k)

            # Fontion quadratique à étudier
            m_k(x) = (f(x) + transpose(g_k) * x + 0.5 * transpose(x) * H_k * x)

            if norm(g_k) < eps()
                continuer = false
                flag = 0
                break
            end

            # Calcul du pas de Cauchy
            s_k, etat_cauchy = Pas_De_Cauchy(g_k, H_k, delta_k)
            
            # println("Pas de chauchy : ", s_k)
            # println("Région de confiance : ", delta_k)

            # Calcul des conditions de mise à jour de l'itéré courant et de la région de confiance
            f_tmp = f(x_k + s_k)
            p_k = abs((f(x_k) - f_tmp) / (m_k(x_tard) - m_k(s_k)))
            # println("p_k = ", p_k)

            # Les deux structures conditionnelles ci-dessous sont séparées par soucis de clarté

            # Mise à jour de l'itéré courant x_k
            if p_k >= eta1
            #    println("true")

                # On sauvegarde x_k uniquement si il change
                x_k_prec = x_k
                val_prec = f(x_k)
                x_k = x_k + s_k
                # println("x change : ", x_k)
            end

            # Mise à jour de la région de confiance
            if p_k >= eta2
                delta_k = min(gamma2 * delta_k, deltaMax)
            else
                if p_k < eta1
                    delta_k = gamma1 * delta_k
                end
            end


            # Mise à jour de val
            val = f(x_k)
            
            nb_iters = nb_iters + 1

            # Tests d'arrêts, séparés pour plus de clarté également

            # Si x_k a changé, on peut le comparer avec l'ancien
            if p_k >= eta1

                norm_sk = norm(s_k)
                if (norm_sk < Tol_abs || norm_sk / min(norm(val), norm(val_prec)) < Tol_rel)
                    continuer = false
                    flag = 2
                end

                if (norm_sk < Tol_abs || norm_sk / min(norm(x_k), norm(x_k_prec)) < Tol_rel)
                    continuer = false
                    flag = 1
                end

            end

            if nb_iters >= max_iter
                continuer = false
                flag = 3
            end

        end

        xmin = x_k
        fxmin = val


    else
        println("rien")
    end
        

    return xmin, fxmin, flag, nb_iters
end
