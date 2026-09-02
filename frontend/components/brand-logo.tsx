import Image from "next/image";
import logo from "../logo.png";

type BrandLogoProps = {
  className?: string;
};

export function BrandLogo({ className = "" }: BrandLogoProps) {
  return (
    <span className="brand-logo-frame" aria-label="PreSense">
      <Image
        src={logo}
        alt=""
        className={`brand-logo ${className}`.trim()}
        priority
      />
    </span>
  );
}
