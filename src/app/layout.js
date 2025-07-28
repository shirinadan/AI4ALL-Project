export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body suppressHydrationWarning={true}>
        <header>BizLens</header>
        {children}
        {/* <footer>My Footer</footer> */}
      </body>
    </html>
  )
}