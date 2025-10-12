/** @type {import('next').NextConfig} */
const nextConfig = {
  trailingSlash: true,
  images: {
    unoptimized: true
  },
  assetPrefix: '/',
  basePath: '',
}

module.exports = nextConfig
