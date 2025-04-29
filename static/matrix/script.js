// Add 3D tilt effect to cards
document.addEventListener('DOMContentLoaded', function() {
    const cards = document.querySelectorAll('.card, .feature-card, .matrix-input-container, .result-container');
    
    cards.forEach(card => {
        card.addEventListener('mousemove', function(e) {
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            const angleX = (y - centerY) / 20;
            const angleY = (centerX - x) / 20;
            
            this.style.transform = `perspective(1000px) rotateX(${angleX}deg) rotateY(${angleY}deg)`;
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'perspective(1000px) rotateX(0) rotateY(0)';
        });
    });
    
    // Add validation for matrix dimensions
    const dimensionInputs = document.querySelectorAll('.dimension-inputs input');
    dimensionInputs.forEach(input => {
        input.addEventListener('input', function() {
            if (parseInt(this.value) < 1) {
                this.value = 1;
            } else if (parseInt(this.value) > 10) {
                this.value = 10;
            }
        });
    });
    
    // Handle matrix operation form submission
    const matrixForms = document.querySelectorAll('.matrix-form');
    matrixForms.forEach(form => {
        form.addEventListener('submit', function(e) {
            // Add loading animation
            const calculationBtn = form.querySelector('.calculate-btn');
            if (calculationBtn) {
                calculationBtn.innerHTML = 'Calculating...';
                calculationBtn.disabled = true;
                
                // Re-enable button after form is submitted
                setTimeout(() => {
                    calculationBtn.innerHTML = 'Calculate';
                    calculationBtn.disabled = false;
                }, 1000);
            }
        });
    });
    
    // Handle matrix cell navigation with arrow keys
    const matrixInputs = document.querySelectorAll('.matrix-table input');
    matrixInputs.forEach(input => {
        input.addEventListener('keydown', function(e) {
            if (e.key === 'ArrowRight' || e.key === 'Tab') {
                const nextInput = this.parentElement.nextElementSibling?.querySelector('input');
                if (nextInput) {
                    nextInput.focus();
                    if (e.key === 'Tab') {
                        e.preventDefault();
                    }
                } else {
                    const nextRow = this.parentElement.parentElement.nextElementSibling;
                    if (nextRow) {
                        const firstInput = nextRow.querySelector('input');
                        if (firstInput) {
                            firstInput.focus();
                            if (e.key === 'Tab') {
                                e.preventDefault();
                            }
                        }
                    }
                }
            } else if (e.key === 'ArrowLeft') {
                const prevInput = this.parentElement.previousElementSibling?.querySelector('input');
                if (prevInput) {
                    prevInput.focus();
                }
            } else if (e.key === 'ArrowDown') {
                const currentRow = this.parentElement.parentElement;
                const nextRow = currentRow.nextElementSibling;
                if (nextRow) {
                    const cellIndex = Array.from(currentRow.children).indexOf(this.parentElement);
                    const nextCell = nextRow.children[cellIndex];
                    if (nextCell) {
                        const nextInput = nextCell.querySelector('input');
                        if (nextInput) {
                            nextInput.focus();
                        }
                    }
                }
            } else if (e.key === 'ArrowUp') {
                const currentRow = this.parentElement.parentElement;
                const prevRow = currentRow.previousElementSibling;
                if (prevRow) {
                    const cellIndex = Array.from(currentRow.children).indexOf(this.parentElement);
                    const prevCell = prevRow.children[cellIndex];
                    if (prevCell) {
                        const prevInput = prevCell.querySelector('input');
                        if (prevInput) {
                            prevInput.focus();
                        }
                    }
                }
            }
        });
    });
    
    // Removed all identity matrix and zero matrix related code
    // This section previously contained the quick fill buttons code which has been removed
});