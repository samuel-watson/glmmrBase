#pragma once

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include "covariance.hpp"

namespace glmmr {

using namespace Eigen;

class spdeCovariance : public Covariance {
public:
  // --- data ---
  int       dim;             // spatial dimension (must be 2 for this prototype)
  int       alpha = 2;       // SPDE order; α=2 ⇒ ν = α - d/2 = 1 for d=2
  int       nv = 0;          // number of mesh vertices
  bool      spde_loaded = false;
  
  // Fixed building blocks (set once by spde_data(), never mutated thereafter)
  SparseMatrix<double> A_;    // n_locations × n_v   observation projector
  VectorXd             C_;    // n_v                 lumped mass (diagonal entries)
  VectorXd             C_inv_;// n_v                 1 / C_
  SparseMatrix<double> G_;    // n_v × n_v           FEM stiffness
  SparseMatrix<double> M_;    // n_v × n_v           G * diag(1/C) * G   (cached)
  SparseMatrix<double> ZA_;   // matZ * A_loc, cached
  
  // Current precision Q = aC * C + aG * G + aM * M   (C and M as sparse mats)
  SparseMatrix<double> C_sp_; // sparse wrapper for C_ (diagonal)
  SparseMatrix<double> Q_mat;
  double aC = 0.0, aG = 0.0, aM = 0.0;
  
  // Cached factorisation — NaturalOrdering for prototype (no permutation to track)
  SimplicialLLT<SparseMatrix<double>, Lower, AMDOrdering<int>> chol_Q;
  bool chol_Q_symbolic_done = false;
  bool chol_Q_current = false;
  
  // Selected inverse of Q on pattern S(L + L^T), filled by Takahashi on demand
  SparseMatrix<double> Sigma_sel;
  bool sel_inv_current = false;
  
  // --- constructors (mirror hsgpCovariance exactly) ---
  spdeCovariance(const std::string& formula,const ArrayXXd& data,const strvec& colnames);
  spdeCovariance(const glmmr::Formula& formula,const ArrayXXd& data,const strvec& colnames);
  spdeCovariance(const std::string& formula,const ArrayXXd& data,const strvec& colnames,const dblvec& parameters);
  spdeCovariance(const glmmr::Formula& formula,const ArrayXXd& data,const strvec& colnames,const dblvec& parameters);
  spdeCovariance(const glmmr::spdeCovariance& cov);
  
  // --- SPDE data loader (called from R after construction, analogous to update_approx_parameters) ---
  void spde_data(const SparseMatrix<double>& A_in,
                 const VectorXd&             C_diag_in,
                 const SparseMatrix<double>& G_in,
                 int                         alpha_in = 2);
  
  // --- virtual overrides ---
  MatrixXd               ZL() override;
  MatrixXd               D(bool chol = true, bool upper = false) override;
  MatrixXd               ZLu(const MatrixXd& u) override;
  MatrixXd               Lu(const MatrixXd& u) override;
  VectorXd               sim_re() override;
  MatrixXd               solve(const MatrixXd& u) override;
  void                   nr_step(const MatrixXd& umat, const MatrixXd& vmat,
                                 ArrayXd& logl, ArrayXd& gradients,
                                 const ArrayXd& uweight) override;
  SparseMatrix<double>   ZL_sparse_new() override;
  int                    Q() const override;
  double                 log_likelihood(const VectorXd& u) override;
  double                 log_determinant() override;
  void                   update_parameters(const dblvec& parameters) override;
  void                   update_parameters(const ArrayXd& parameters) override;
  void                   update_parameters_extern(const dblvec& parameters) override;
  int                    npar() const override;
  intvec                 parameter_fn_index() const override;
  
  // --- SPDE-specific machinery used by the upstream nr_theta / posterior_u_sample branches ---
  dblvec                 get_parameters_extern() const;
  void                   build_Q();              // assembles Q_mat from (σ², λ)
  void                   refactor_Q();           // numeric Cholesky (symbolic once)
  void                   compute_selected_inverse();  // Takahashi on pattern S(L+L^T)
  void                   refresh_probes();
  // Sparse quadratic-form helpers (use existing factorisation / building blocks)
  double                 quad_form_Q(const VectorXd& u);              // u^T Q u
  double                 quad_form_C(const VectorXd& u);              // u^T C u
  double                 quad_form_M(const VectorXd& u);              // u^T M u
  double                 quad_form_dQ_log_sigmasq(const VectorXd& u); // u^T ∂Q/∂log σ² u  = -u^T Q u
  double                 quad_form_dQ_log_lambda(const VectorXd& u);  // u^T ∂Q/∂log λ u
  double                 trace_Qinv_dQ_Qinv_dQ_lambda(int K = 30);
  double                 trace_Qinv_dQ_log_lambda_hutch(int K = 30);
  std::pair<double,double> traces_for_lambda(int K = 30);
  // Exact trace tr(Q^{-1} ∂Q/∂log λ) via selected inverse; σ² version is analytic (-n_v)
  double                 trace_Qinv_dQ_log_lambda();
  
  // Posterior-precision factorisation P = A^T diag(W) A + Q — cached separately
  SimplicialLLT<SparseMatrix<double>, Lower, AMDOrdering<int>> chol_P;
  bool                   chol_P_symbolic_done = false;
  void                   refactor_P(const VectorXd& W_diag);  // for posterior_u_sample
  
protected:
  void  parse_spde_term();       // inspect formula, identify spde(x, y) entry
  void  update_coefficients();   // (σ², λ) → (aC, aG, aM)
  void  mark_stale();            // invalidate chol_Q, Sigma_sel
  std::vector<VectorXd> probe_cache;
  int probe_K = 50;
  bool probes_current = false;
  
};

// ============================================================================
// Constructors
// ============================================================================

inline glmmr::spdeCovariance::spdeCovariance(const std::string& formula,
                                             const ArrayXXd& data,
                                             const strvec& colnames)
  : Covariance(formula, data, colnames),
    dim(this->re_cols_data_[0][0].size())
{
  isSparse = true;          // native sparse — downstream can branch on this
  parse_spde_term();
  // nv and all building blocks remain unset until spde_data() is called.
}

inline glmmr::spdeCovariance::spdeCovariance(const glmmr::Formula& formula,
                                             const ArrayXXd& data,
                                             const strvec& colnames)
  : Covariance(formula, data, colnames),
    dim(this->re_cols_data_[0][0].size())
{
  isSparse = true;
  parse_spde_term();
}

inline glmmr::spdeCovariance::spdeCovariance(const std::string& formula,
                                             const ArrayXXd& data,
                                             const strvec& colnames,
                                             const dblvec& parameters)
  : Covariance(formula, data, colnames, parameters),
    dim(this->re_cols_data_[0][0].size())
{
  isSparse = true;
  parse_spde_term();
  // Cannot build Q yet — need spde_data() first. User must call it before any fitting op.
}

inline glmmr::spdeCovariance::spdeCovariance(const glmmr::Formula& formula,
                                             const ArrayXXd& data,
                                             const strvec& colnames,
                                             const dblvec& parameters)
  : Covariance(formula, data, colnames, parameters),
    dim(this->re_cols_data_[0][0].size())
{
  isSparse = true;
  parse_spde_term();
}

inline glmmr::spdeCovariance::spdeCovariance(const glmmr::spdeCovariance& cov)
  : Covariance(cov.form_, cov.data_, cov.colnames_, cov.parameters_),
    dim(cov.dim), alpha(cov.alpha), nv(cov.nv), spde_loaded(cov.spde_loaded),
    A_(cov.A_), C_(cov.C_), C_inv_(cov.C_inv_), G_(cov.G_), M_(cov.M_),
    C_sp_(cov.C_sp_), Q_mat(cov.Q_mat),
    aC(cov.aC), aG(cov.aG), aM(cov.aM)
{
  isSparse = true;
  chol_Q_symbolic_done = false;    // factorisations are non-copyable; rebuild lazily
  chol_P_symbolic_done = false;
  chol_Q_current = false;
  sel_inv_current = false;
  if(spde_loaded && parameters_.size() >= 2){
    build_Q();
    refactor_Q();
  }
}

// ============================================================================
// Formula parsing — identify the spde(...) term, extract coordinate columns
// ============================================================================

inline void glmmr::spdeCovariance::parse_spde_term(){
  // Base-class parser has already run and populated re_cols_data_, fn_, etc.
  // We just validate what we got.
  if(dim != 2){
    throw std::runtime_error("spdeCovariance prototype supports dim = 2 only.");
  }
  if(this->fn_.empty() || this->fn_[0].empty()){
    throw std::runtime_error("spdeCovariance: no covariance function detected by parser.");
  }
  // For the prototype, accept any Matérn-family tag — ν is fixed by alpha=2.
  const auto& fn0 = this->fn_[0][0];
  bool recognised = (fn0 == CovFunc::matern  || fn0 == CovFunc::matern1log ||
                     fn0 == CovFunc::matern2log ||
                     fn0 == CovFunc::fexp   || fn0 == CovFunc::fexp0    ||
                     fn0 == CovFunc::fexplog);
  if(!recognised){
    throw std::runtime_error("spdeCovariance: formula must use a Matérn-family function "
                               "(e.g. spde_matern1log(x, y)).");
  }
}

// ============================================================================
// SPDE data loader
// ============================================================================

inline void glmmr::spdeCovariance::spde_data(const SparseMatrix<double>& A_in,
                                             const VectorXd&             C_diag_in,
                                             const SparseMatrix<double>& G_in,
                                             int                         alpha_in)
{
  if(alpha_in != 2){
    throw std::runtime_error("spdeCovariance prototype supports alpha = 2 only (ν = 1 in 2D).");
  }
  if(G_in.rows() != G_in.cols() || G_in.rows() != C_diag_in.size()){
    throw std::runtime_error("spde_data: C and G dimensions inconsistent.");
  }
  if(A_in.cols() != G_in.rows()){
    throw std::runtime_error("spde_data: A columns must equal mesh vertex count.");
  }
  if(A_in.rows() != this->matZ.cols()){
    throw std::runtime_error("spde_data: A must have one row per unique location "
                               "(rows=" + std::to_string(A_in.rows()) +
                                 ", expected=" + std::to_string(this->matZ.cols()) + ").");
  }
  
  alpha = alpha_in;
  nv    = G_in.rows();
  A_ = A_in;
  C_    = C_diag_in;
  C_inv_.resize(nv);
  for(int i = 0; i < nv; ++i){
    if(C_(i) <= 0) throw std::runtime_error("spde_data: lumped mass C has non-positive entry.");
    C_inv_(i) = 1.0 / C_(i);
  }
  G_ = G_in;
  
  // Build C as sparse (diagonal) for convenient assembly of Q
  C_sp_.resize(nv, nv);
  {
    std::vector<Triplet<double>> trips; trips.reserve(nv);
    for(int i = 0; i < nv; ++i) trips.emplace_back(i, i, C_(i));
    C_sp_.setFromTriplets(trips.begin(), trips.end());
  }
 
  // M = G * diag(1/C) * G  (cached; nonzero pattern is G*G up to sparsity of G)
  SparseMatrix<double> Cinv_sp(nv, nv);
  {
    std::vector<Triplet<double>> trips; trips.reserve(nv);
    for(int i = 0; i < nv; ++i) trips.emplace_back(i, i, C_inv_(i));
    Cinv_sp.setFromTriplets(trips.begin(), trips.end());
  }
  M_ = G_ * Cinv_sp * G_;
  
  // Override the base-class Q_ (= number of random effects); we are n_v, not matZ.cols()
  this->Q_ = nv;
  
  spde_loaded = true;
  chol_Q_symbolic_done = false;
  chol_Q_current       = false;
  chol_P_symbolic_done = false;
  sel_inv_current      = false;
  
  ZA_ = this->matZ * A_;
}

// ============================================================================
// Coefficients, precision assembly, factorisation
// ============================================================================

inline void glmmr::spdeCovariance::refresh_probes(){
  thread_local std::mt19937 gen{42};  // fixed seed across calls within an iteration
  std::normal_distribution<double> nrm(0.0, 1.0);
  probe_cache.resize(probe_K);
  for(auto& z : probe_cache){
    z.resize(nv);
    for(int i = 0; i < nv; ++i) z(i) = nrm(gen);
  }
  probes_current = true;
}

inline void glmmr::spdeCovariance::update_coefficients(){
  // Maps (σ², λ) to the three scalar coefficients of Q = aC C + aG G + aM M
  // for α = 2, ν = 1, d = 2. κ = sqrt(8ν)/λ = 2√2 / λ, τ² from INLA formula.
  bool logpars = all_log_re();
  double sigma2 = logpars ? std::exp(parameters_[0]) : parameters_[0];
  double lambda = logpars ? std::exp(parameters_[1]) : parameters_[1];
  
  // Q = τ² ( κ⁴ C + 2 κ² G + G C⁻¹ G )
  //   = (1/(4π σ² κ²))  * ( κ⁴ C + 2 κ² G + M )
  //   = (κ² / (4π σ²)) C + (1/(2π σ²)) G + (1/(4π σ² κ²)) M
  //
  // With κ² = 8/λ²:
  //   aC = (8/λ²) / (4π σ²)                = 2 / (π σ² λ²)
  //   aG =  1    / (2π σ²)
  //   aM = (λ²/8)/ (4π σ²)                 = λ² / (32 π σ²)
  const double pi = M_PI;
  aC = 2.0                  / (pi * sigma2 * lambda * lambda);
  aG = 1.0                  / (2.0 * pi * sigma2);
  aM = (lambda * lambda)    / (32.0 * pi * sigma2);
}

inline void glmmr::spdeCovariance::build_Q(){
  if(!spde_loaded) throw std::runtime_error("spdeCovariance: spde_data() must be called first.");
  update_coefficients();
  Q_mat = aC * C_sp_ + aG * G_ + aM * M_;  // sparse + sparse
  chol_Q_current  = false;
  sel_inv_current = false;
}

inline void glmmr::spdeCovariance::refactor_Q(){
  if(!spde_loaded) throw std::runtime_error("spdeCovariance: spde_data() must be called first.");
  if(!chol_Q_symbolic_done){
    chol_Q.analyzePattern(Q_mat);
    chol_Q_symbolic_done = true;
  }
  chol_Q.factorize(Q_mat);
  if(chol_Q.info() != Success){
    throw std::runtime_error("spdeCovariance: Cholesky factorisation of Q failed.");
  }
  chol_Q_current  = true;
  sel_inv_current = false;
}

inline void glmmr::spdeCovariance::mark_stale(){
  chol_Q_current  = false;
  sel_inv_current = false;
}

// ============================================================================
// Virtual overrides — the simple ones
// ============================================================================

inline int glmmr::spdeCovariance::Q() const { return this->Q_; }   // = nv after spde_data

inline int glmmr::spdeCovariance::npar() const { return 2; }       // σ², λ — isotropic only

inline intvec glmmr::spdeCovariance::parameter_fn_index() const {
  return re_fn_par_link_;    // base-class default; adjust later if anisotropy added
}

inline void glmmr::spdeCovariance::update_parameters(const dblvec& parameters){
  parameters_ = parameters;
  if(spde_loaded){ build_Q(); refactor_Q(); }
}

inline void glmmr::spdeCovariance::update_parameters(const ArrayXd& parameters){
  if(parameters_.size() == 0){
    for(int i = 0; i < parameters.size(); ++i) parameters_.push_back(parameters(i));
  } else {
    for(int i = 0; i < parameters.size(); ++i) parameters_[i] = parameters(i);
  }
  if(spde_loaded){ build_Q(); refactor_Q(); }
}

inline void glmmr::spdeCovariance::update_parameters_extern(const dblvec& parameters){
  if(static_cast<int>(parameters.size()) != npar()){
    throw std::runtime_error("spdeCovariance: wrong number of parameters.");
  }
  // No coordinate rescaling in the prototype — user's mesh scale is authoritative.
  parameters_ = parameters;
  if(spde_loaded){ build_Q(); refactor_Q(); }
}

inline dblvec glmmr::spdeCovariance::get_parameters_extern() const {
  return parameters_;
}

// ============================================================================
// Sampling, likelihood, logdet
// ============================================================================

inline VectorXd glmmr::spdeCovariance::sim_re(){
  if(!spde_loaded)               throw std::runtime_error("spdeCovariance: no mesh data.");
  if(parameters_.size() == 0)    throw std::runtime_error("spdeCovariance: no parameters.");
  if(!chol_Q_current)            refactor_Q();
  
  std::random_device rd{};
  std::mt19937 gen{ rd() };
  std::normal_distribution<double> d{ 0.0, 1.0 };
  VectorXd z(nv);
  for(int j = 0; j < nv; ++j) z(j) = d(gen);
  
  // u ~ N(0, Q^{-1}). With Q = L L^T (NaturalOrdering, no permutation),
  // u = L^{-T} z satisfies Cov(u) = L^{-T} L^{-1} = Q^{-1}.
  // matrixU() returns L^T for SimplicialLLT.
  VectorXd u_perm = chol_Q.matrixU().solve(z);
  return chol_Q.permutationP().transpose() * u_perm;
}

inline double glmmr::spdeCovariance::log_likelihood(const VectorXd& u){
  // log p(u) = -(n_v/2) log(2π) + (1/2) log|Q| - (1/2) u^T Q u
  static const double LOG_2PI = std::log(2.0 * M_PI);
  if(!chol_Q_current) refactor_Q();
  // log|Q| = 2 * sum(log(diag(L))) for Q = L L^T
  double logdet_Q = 2.0 * chol_Q.matrixL().nestedExpression().diagonal().array().log().sum();
  double qf       = quad_form_Q(u);
  return -0.5 * static_cast<double>(nv) * LOG_2PI + 0.5 * logdet_Q - 0.5 * qf;
}

inline double glmmr::spdeCovariance::log_determinant(){
  // Consistent with hsgp: returns log|D| where D is the prior cov of u.
  // For SPDE D = Q^{-1}, so log|D| = -log|Q|.
  if(!chol_Q_current) refactor_Q();
  double logdet_Q = 2.0 * chol_Q.matrixL().nestedExpression().diagonal().array().log().sum();
  return -logdet_Q;
}

// ============================================================================
// Sparse operations used by upstream branches
// ============================================================================

inline double glmmr::spdeCovariance::quad_form_Q(const VectorXd& u){
  return (u.transpose() * Q_mat.selfadjointView<Lower>() * u).value();
}
inline double glmmr::spdeCovariance::quad_form_C(const VectorXd& u){
  return u.cwiseProduct(C_).dot(u);   // diagonal
}
inline double glmmr::spdeCovariance::quad_form_M(const VectorXd& u){
  return (u.transpose() * M_.selfadjointView<Lower>() * u).value();
}

inline double glmmr::spdeCovariance::quad_form_dQ_log_sigmasq(const VectorXd& u){
  // ∂Q/∂log σ² = -Q   (every coefficient of Q has a 1/σ² factor)
  return -quad_form_Q(u);
}
inline double glmmr::spdeCovariance::quad_form_dQ_log_lambda(const VectorXd& u){
  // ∂aC/∂log λ = -2 aC,  ∂aG/∂log λ = 0,  ∂aM/∂log λ = +2 aM
  return -2.0 * aC * quad_form_C(u) + 2.0 * aM * quad_form_M(u);
}

inline double glmmr::spdeCovariance::trace_Qinv_dQ_Qinv_dQ_lambda(int K){
  if(!chol_Q_current) refactor_Q();
  SparseMatrix<double> Ql = (-2.0 * aC) * C_sp_ + (2.0 * aM) * M_;
  // tr(Q⁻¹ Qλ Q⁻¹ Qλ) = E[z^T Qλ Q⁻¹ Qλ Q⁻¹ z]  for z ~ N(0, I) (or Rademacher)
  thread_local std::mt19937 gen{std::random_device{}()};
  std::normal_distribution<double> nrm(0.0, 1.0);
  double acc = 0.0;
  VectorXd z(nv), a(nv), b(nv), c(nv);
  for(int k = 0; k < K; ++k){
    for(int i = 0; i < nv; ++i) z(i) = nrm(gen);
    a = chol_Q.solve(z);    // Q⁻¹ z
    b = Ql * a;             // Qλ Q⁻¹ z
    c = chol_Q.solve(b);    // Q⁻¹ Qλ Q⁻¹ z
    acc += (Ql * c).dot(z); // z^T Qλ (...) 
  }
  return acc / static_cast<double>(K);
}

inline double glmmr::spdeCovariance::trace_Qinv_dQ_log_lambda_hutch(int K){
  if(!chol_Q_current) refactor_Q();
  SparseMatrix<double> Ql = (-2.0 * aC) * C_sp_ + (2.0 * aM) * M_;
  thread_local std::mt19937 gen{std::random_device{}()};
  std::normal_distribution<double> nrm(0.0, 1.0);
  double acc = 0.0;
  VectorXd z(nv), a(nv);
  for(int k = 0; k < K; ++k){
    for(int i = 0; i < nv; ++i) z(i) = nrm(gen);
    a = chol_Q.solve(z);        // Q⁻¹ z
    acc += (Ql * a).dot(z);     // z^T Qλ Q⁻¹ z
  }
  return acc / static_cast<double>(K);
}

inline std::pair<double,double> 
  glmmr::spdeCovariance::traces_for_lambda(int K){
    if(!chol_Q_current) refactor_Q();
    if(!probes_current) refresh_probes();
    
    double acc_grad = 0.0, acc_hess = 0.0;
    VectorXd a(nv), b(nv), c(nv);
    
    for(int k = 0; k < K; ++k){
      const VectorXd& z = probe_cache[k];
      a = chol_Q.solve(z);                          // Q⁻¹ z  (1 sparse solve)
      b.noalias() = -2.0*aC * (C_sp_ * a) + 2.0*aM * (M_ * a);  // Qλ Q⁻¹ z
      acc_grad += b.dot(z);                         // gradient trace
      c = chol_Q.solve(b);                          // Q⁻¹ Qλ Q⁻¹ z (2nd solve)
      VectorXd d = -2.0*aC * (C_sp_ * c) + 2.0*aM * (M_ * c);   // Qλ Q⁻¹ Qλ Q⁻¹ z
      acc_hess += d.dot(z);                         // Hessian trace
    }
    return {acc_grad / K, acc_hess / K};
  }

inline MatrixXd glmmr::spdeCovariance::ZLu(const MatrixXd& u){
  // u is n_v × m (m MC samples). Result is n_obs × m.
  // (sparse * sparse) * dense is fine; sparse * (sparse * dense) = sparse * dense also fine.
  return this->matZ * (A_ * u);
}

inline MatrixXd glmmr::spdeCovariance::Lu(const MatrixXd& u){
  return A_ * u;
}

inline MatrixXd glmmr::spdeCovariance::solve(const MatrixXd& u){
  // For HSGP, solve returns D^{-1} u = u / Lambda. For SPDE D = Q^{-1}, so D^{-1} u = Q u.
  return MatrixXd(Q_mat.selfadjointView<Lower>() * u);
}

inline SparseMatrix<double> glmmr::spdeCovariance::ZL_sparse_new(){
  return this->matZ * A_;   // n_obs × n_v, 3 nonzeros per row
}
// ============================================================================
// Posterior-precision factorisation for posterior_u_sample
// ============================================================================

inline void glmmr::spdeCovariance::refactor_P(const VectorXd& W_diag){
  SparseMatrix<double> P = ZA_.transpose() * W_diag.asDiagonal() * ZA_ + Q_mat;
  if(!chol_P_symbolic_done){
    chol_P.analyzePattern(P);
    chol_P_symbolic_done = true;
  }
  chol_P.factorize(P);
  if(chol_P.info() != Success){
    throw std::runtime_error("spdeCovariance: Cholesky factorisation of P failed.");
  }
}

// ============================================================================
// Selected inverse — Takahashi recursion on pattern S(L + L^T)
// ============================================================================

inline void glmmr::spdeCovariance::compute_selected_inverse(){
  throw std::runtime_error("compute_selected_inverse: not supported under AMD ordering. "
                             "Use Hutchinson-based trace estimators instead.");
}
// ============================================================================
// Trace tr(Q^{-1} ∂Q/∂log λ) — uses Sigma_sel.
// ============================================================================

inline double glmmr::spdeCovariance::trace_Qinv_dQ_log_lambda(){
  // ∂Q/∂log λ = -2 aC · C + 2 aM · M.
  // tr(Q^{-1} · C) = Σ_i Σ_sel_{ii} C_i     (C diagonal)
  // tr(Q^{-1} · M) = Σ_{(i,j) ∈ S(M)} Σ_sel_{ij} M_{ji}
  if(!sel_inv_current) compute_selected_inverse();
  
  double tr_QinvC = 0.0;
  for(int i = 0; i < nv; ++i){
    tr_QinvC += Sigma_sel.coeff(i, i) * C_(i);
  }
  
  double tr_QinvM = 0.0;
  for(int col = 0; col < M_.outerSize(); ++col){
    for(SparseMatrix<double>::InnerIterator it(M_, col); it; ++it){
      // S(M) is expected ⊆ S(L + L^T) for typical 2D FEM meshes.
      // If not, Sigma_sel.coeff(...) returns 0, under-counting the trace.
      // Worth a runtime check in production; skipped here for brevity.
      tr_QinvM += it.value() * Sigma_sel.coeff((int)it.row(), (int)it.col());
    }
  }
  
  return -2.0 * aC * tr_QinvC + 2.0 * aM * tr_QinvM;
}

// ============================================================================
// Deliberate-throw overrides — upstream must branch on covariance type before calling
// ============================================================================

inline MatrixXd glmmr::spdeCovariance::ZL(){
  throw std::runtime_error("spdeCovariance::ZL() — Q^{-1} is dense; upstream must branch on type.");
}
inline MatrixXd glmmr::spdeCovariance::D(bool chol, bool upper){
  if(!spde_loaded) throw std::runtime_error("spdeCovariance::D(): spde_data() not loaded.");
  if(!chol_Q_current) refactor_Q();
  
  const int n = nv;
  
  if(chol){
    // chol_Q factorises P_sym · Q · P_sym^T = R^T R, with R = chol_Q.matrixU().
    // Covariance under u = P_sym^T · x, where x ~ N(0, (P_sym Q P_sym^T)^{-1}):
    //   Cov(u) = P_sym^T · (R^T R)^{-1} · P_sym = Q^{-1}.
    // So a valid factor is L_u = P_sym^T · R^{-1}, giving Cov(u) = L_u L_u^T = Q^{-1}.
    MatrixXd I = MatrixXd::Identity(n, n);
    MatrixXd R_inv = chol_Q.matrixU().solve(I);  // R^{-1}, upper-triangular in permuted space
    // Un-permute rows to return factor in original-vertex space
    MatrixXd L_u = chol_Q.permutationP().transpose() * R_inv;
    
    if(upper){
      // Caller wants U such that D = U^T U. From D = L_u L_u^T, U = L_u^T.
      return L_u.transpose();
    } else {
      // Caller wants L such that D = L L^T.
      return L_u;
    }
  } else {
    // Dense Q^{-1}: permutation handled internally by solve().
    return chol_Q.solve(MatrixXd::Identity(n, n));
  }
}
inline void glmmr::spdeCovariance::nr_step(const MatrixXd& /*umat*/, const MatrixXd& /*vmat*/,
                                           ArrayXd& /*logl*/, ArrayXd& /*gradients*/,
                                           const ArrayXd& /*uweight*/)
{
  throw std::runtime_error("spdeCovariance::nr_step — use the nr_theta SPDE specialisation in modeloptim.hpp.");
}

}  // namespace glmmr