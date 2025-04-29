document.addEventListener('DOMContentLoaded', () => {
    // Calculus card hover effects
    const calculusCards = document.querySelectorAll('.calculus-card');
    calculusCards.forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'translateY(-5px)';
            card.style.boxShadow = '0 10px 25px rgba(52, 152, 219, 0.4)';
        });
        card.addEventListener('mouseleave', () => {
            card.style.transform = '';
            card.style.boxShadow = '0 4px 15px rgba(0, 0, 0, 0.2)';
        });
    });

    // Form input effects
    const inputs = document.querySelectorAll('input, textarea, select');
    inputs.forEach(input => {
        input.addEventListener('focus', () => {
            input.style.boxShadow = '0 0 0 0.25rem rgba(52, 152, 219, 0.25)';
        });
        input.addEventListener('blur', () => {
            input.style.boxShadow = '';
        });
    });

    // Math rendering initialization
    if (typeof MathJax !== 'undefined') {
        MathJax.Hub.Config({
            tex2jax: {
                inlineMath: [['$','$'], ['\\(','\\)']],
                processEscapes: true
            }
        });
    }
});