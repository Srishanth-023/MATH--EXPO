document.addEventListener('DOMContentLoaded', () => {
    // Create canvas
    const canvas = document.createElement('canvas');
    canvas.id = 'animationCanvas';
    document.body.appendChild(canvas);
    const ctx = canvas.getContext('2d');

    // Resize
    const resize = () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    // Sine wave animation
    let t = 0;
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw sine wave
        ctx.beginPath();
        ctx.moveTo(0, canvas.height / 2);
        for (let x = 0; x < canvas.width; x++) {
            const y = Math.sin((x + t) * 0.02) * 50 + canvas.height / 2;
            ctx.lineTo(x, y);
        }
        ctx.strokeStyle = 'rgba(0, 150, 136, 0.7)';
        ctx.lineWidth = 2;
        ctx.stroke();

        t += 2;
        requestAnimationFrame(animate);
    }
    animate();

    // Floating trig labels (optional)
    const labels = ["sin(θ)", "cos(θ)", "tan(θ)", "π", "90°", "θ", "csc(θ)", "sec(θ)", "cot(θ)"];
    labels.forEach((text, index) => {
        const div = document.createElement('div');
        div.className = 'trig-label';
        div.style.top = `${Math.random() * 90 + 5}%`;
        div.style.left = `${Math.random() * 90 + 5}%`;
        div.style.animationDelay = `${Math.random() * 5}s`;
        div.innerText = text;
        document.body.appendChild(div);
    });
});

// Animate cards on load using GSAP
window.addEventListener('DOMContentLoaded', () => {
    gsap.from(".animated-grid .col-sm-6", {
        duration: 1,
        opacity: 0,
        y: 40,
        stagger: 0.1,
        ease: "power2.out"
    });
});

// Use GSAP to animate the exotic trigonometric elements after the page loads
document.addEventListener('DOMContentLoaded', function() {
    // Animate the circle rotation
    gsap.from('.circle-animate', {
        opacity: 0,
        scale: 0.5,
        duration: 1.5,
        delay: 0.5,
        ease: "power4.out"
    });

    // Animate the wave motion with a subtle delay
    gsap.from('.wave-animation', {
        opacity: 0,
        x: -200,
        duration: 2,
        delay: 1,
        ease: "power2.out"
    });

    // Animate the triangle rotation
    gsap.from('.triangle-rotate', {
        opacity: 0,
        rotation: -180,
        duration: 1.5,
        delay: 1.5,
        ease: "power2.inOut"
    });
});

