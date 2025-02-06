const hamburger = document.querySelector('.hamburger');
        const navLinks = document.querySelector('.nav-links');
        const overlay = document.querySelector('.overlay');
        const body = document.body;

        // Toggle menu
        hamburger.addEventListener('click', () => {
            toggleMenu();
        });

        // Close menu when clicking overlay
        overlay.addEventListener('click', () => {
            toggleMenu();
        });

        // Close menu when clicking nav links
        document.querySelectorAll('.nav-links a').forEach(link => {
            link.addEventListener('click', () => {
                toggleMenu();
            });
        });

        // Function to toggle menu state
        function toggleMenu() {
            hamburger.classList.toggle('active');
            navLinks.classList.toggle('active');
            overlay.classList.toggle('active');
            body.classList.toggle('menu-open');
        }

        // Navbar scroll effect
        let lastScroll = 0;
        window.addEventListener('scroll', () => {
            const currentScroll = window.pageYOffset;
            const navbar = document.querySelector('.navbar');

            if (currentScroll <= 0) {
                navbar.style.boxShadow = 'none';
            } else {
                navbar.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.1)';
            }

            lastScroll = currentScroll;
        });

        // Button animation
        const konsultasiBtn = document.querySelector('.btn-scan');

        konsultasiBtn.addEventListener('mouseenter', function(e) {
            this.style.transform = 'translateY(-3px) scale(1.05)';
        });

        konsultasiBtn.addEventListener('mouseleave', function(e) {
            this.style.transform = 'translateY(0) scale(1)';
        });