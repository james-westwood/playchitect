import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Playchitect',
  description: 'Smart DJ Playlist Manager with Intelligent BPM Clustering',
  base: '/playchitect/',

  // Exclude non-user-facing content from the build
  srcExclude: ['**/planning/**', '**/research/**', '**/*.html'],

  head: [
    ['link', { rel: 'icon', href: '/playchitect/favicon.ico' }]
  ],

  themeConfig: {
    logo: { src: 'https://raw.githubusercontent.com/james-westwood/playchitect/main/img/playchitect_logo.jpg', alt: 'Playchitect' },

    nav: [
      { text: 'User Guide', link: '/guide/' },
      { text: 'Contributing', link: '/contributing/' },
      {
        text: 'GitHub',
        link: 'https://github.com/james-westwood/playchitect'
      }
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'User Guide',
          items: [
            { text: 'What is Playchitect?', link: '/guide/' },
            { text: 'Installation', link: '/guide/installation' },
            { text: 'Quick Start', link: '/guide/quick-start' },
            { text: 'CLI Reference', link: '/guide/cli-reference' },
          ]
        }
      ],
      '/contributing/': [
        {
          text: 'Contributing',
          items: [
            { text: 'Getting Started', link: '/contributing/' },
            { text: 'Architecture', link: '/contributing/architecture' },
            { text: 'Testing', link: '/contributing/testing' },
            { text: 'Releasing', link: '/contributing/releasing' },
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/james-westwood/playchitect' }
    ],

    search: {
      provider: 'local'
    },

    footer: {
      message: 'Released under the GPL-3.0 License.',
      copyright: 'Copyright © 2025 James Westwood'
    },

    editLink: {
      pattern: 'https://github.com/james-westwood/playchitect/edit/main/docs/:path',
      text: 'Edit this page on GitHub'
    },

    lastUpdated: {
      text: 'Last updated',
    }
  }
})
