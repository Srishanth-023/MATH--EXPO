document.addEventListener('DOMContentLoaded', function() {
    // Auto-format math input
    document.querySelectorAll('input[type="text"]').forEach(input => {
        input.addEventListener('keyup', function(e) {
            if (e.key === 'Enter') {
                this.form.submit();
            }
        });
    });

    // Copy result to clipboard
    document.querySelectorAll('.math-result').forEach(result => {
        result.addEventListener('click', function() {
            const text = this.innerText;
            navigator.clipboard.writeText(text).then(() => {
                const originalText = this.innerText;
                this.innerText = 'Copied!';
                setTimeout(() => {
                    this.innerText = originalText;
                }, 2000);
            });
        });
    });

    // Example click-to-fill
    document.querySelectorAll('.list-group-item').forEach(example => {
        example.addEventListener('click', function() {
            const input = this.closest('.tool-container').querySelector('input[type="text"]');
            input.value = this.querySelector('code').innerText;
            input.focus();
        });
    });
});

document.addEventListener('DOMContentLoaded', function() {
    // Make example buttons work globally
    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const example = this.textContent;
            const form = this.closest('.card').querySelector('form');
            const input = form.querySelector('input[type="text"]');
            input.value = example;
            input.focus();
        });
    });

    // Form submission handling
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', function(e) {
            const submitBtn = this.querySelector('button[type="submit"]');
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
        });
    });
});

document.addEventListener('DOMContentLoaded', function() {
    // Get all elements
    const slopeInput = document.getElementById('slope');
    const interceptInput = document.getElementById('intercept');
    const xInput = document.getElementById('x');
    const clearBtn = document.getElementById('clearBtn');
    const exampleButtons = document.querySelectorAll('.btn-example');
    
    // Clear form function
    function clearForm() {
        slopeInput.value = '';
        interceptInput.value = '';
        xInput.value = '';
    }
    
    // Fill example function
    function fillExample(m, c, x) {
        slopeInput.value = m;
        interceptInput.value = c;
        xInput.value = x;
    }
    
    // Event listeners
    clearBtn.addEventListener('click', clearForm);
    
    exampleButtons.forEach(button => {
        button.addEventListener('click', function() {
            const m = this.dataset.m;
            const c = this.dataset.c;
            const x = this.dataset.x;
            fillExample(m, c, x);
        });
    });
});

document.addEventListener('DOMContentLoaded', function() {
    // Get all elements
    const aInput = document.getElementById('a');
    const bInput = document.getElementById('b');
    const cInput = document.getElementById('c');
    const clearBtn = document.getElementById('clearBtn');
    const exampleButtons = document.querySelectorAll('.btn-example');
    
    // Clear form function
    function clearForm() {
        aInput.value = '';
        bInput.value = '';
        cInput.value = '';
    }
    
    // Fill example function
    function fillExample(a, b, c) {
        aInput.value = a;
        bInput.value = b;
        cInput.value = c;
    }
    
    // Event listeners
    clearBtn.addEventListener('click', clearForm);
    
    exampleButtons.forEach(button => {
        button.addEventListener('click', function() {
            const a = this.dataset.a;
            const b = this.dataset.b;
            const c = this.dataset.c;
            fillExample(a, b, c);
        });
    });
});
document.addEventListener('DOMContentLoaded', function() {
    const baseInput = document.getElementById('base');
    const powerInput = document.getElementById('power');
    const clearBtn = document.getElementById('clearBtn');
    const exampleButtons = document.querySelectorAll('.btn-example');
    const form = document.getElementById('exponentForm');

    // Clear form
    clearBtn.addEventListener('click', function() {
        baseInput.value = '';
        powerInput.value = '';
        baseInput.focus();
    });

    // Fill examples
    exampleButtons.forEach(button => {
        button.addEventListener('click', function() {
            baseInput.value = this.dataset.base;
            powerInput.value = this.dataset.power;
            baseInput.focus();
        });
    });

    // Form validation
    form.addEventListener('submit', function(e) {
        if (!baseInput.value || !powerInput.value) {
            e.preventDefault();
            alert('Please enter both base and power values');
        }
    });
});

document.addEventListener('DOMContentLoaded', function() {
    const numeratorInput = document.getElementById('numerator');
    const denominatorInput = document.getElementById('denominator');
    const clearBtn = document.getElementById('clearBtn');
    const exampleButtons = document.querySelectorAll('.btn-example');
    const form = document.getElementById('rationalForm');

    // Clear form
    clearBtn.addEventListener('click', function() {
        numeratorInput.value = '';
        denominatorInput.value = '';
        numeratorInput.focus();
    });

    // Fill examples
    exampleButtons.forEach(button => {
        button.addEventListener('click', function() {
            numeratorInput.value = this.dataset.num;
            denominatorInput.value = this.dataset.den;
        });
    });

    // Form validation
    form.addEventListener('submit', function(e) {
        if (!numeratorInput.value || !denominatorInput.value) {
            e.preventDefault();
            alert('Please enter both numerator and denominator');
        }
    });

    // Render MathJax when results update
    if (typeof MathJax !== 'undefined') {
        MathJax.typesetPromise();
    }
});